"""
donor_fill_v2.py — Модуль ориентированного донорного заполнения (Шаги 5 и 6 Спецификации v2).
Заполняет точечные провалы M_fail с учетом доминирующего направления градиентов (структурного тензора),
подгонки фазы растра FFT и локального поиска патчей (окно ±50..100px).
"""

import cv2
import numpy as np


def compute_structure_tensor_orientation(gray: np.ndarray, mask_boundary: np.ndarray) -> float:
    """Определяет доминирующее направление градиента по контуру области."""
    if not np.any(mask_boundary):
        return 0.0
        
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    gx = sobelx[mask_boundary]
    gy = sobely[mask_boundary]
    
    j11 = np.sum(gx * gx)
    j22 = np.sum(gy * gy)
    j12 = np.sum(gx * gy)
    
    angle = 0.5 * np.arctan2(2 * j12, j11 - j22)
    return float(angle)


def feather_blend_patch(target: np.ndarray, donor: np.ndarray, mask: np.ndarray, feather_px: int = 4) -> np.ndarray:
    """Плавно смешивает донорный патч с целевой областью с альфа-градиентом 3-5px."""
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    alpha = np.clip(dist / float(max(1, feather_px)), 0.0, 1.0)
    if len(target.shape) == 3:
        alpha = alpha[:, :, np.newaxis]
        
    blended = donor.astype(np.float32) * alpha + target.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def region_needs_texture(image: np.ndarray, mask: np.ndarray, ring_width: int = 15) -> bool:
    """
    Классифицирует тип региона по высокочастотной энергии растра:
    1. Однородный / белая одежда / черная тень -> donor НЕ нужен (оставить 100% LaMa).
    2. Повторяющийся растровый скринтон (halftone dots) -> donor ТРЕБУЕТСЯ для бесшовного растра.
    """
    k_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_width * 2 + 1, ring_width * 2 + 1))
    ring = (cv2.dilate((mask > 0).astype(np.uint8), k_ring) > 0) & (mask == 0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    ring_pixels = gray[ring]
    
    if ring_pixels.size == 0:
        return False
        
    ring_std = float(ring_pixels.std())
    ring_mean = float(ring_pixels.mean())
    
    # Защита белых областей (>240), черных теней (<15) и гладких градиентов (std <= 8.0)
    if ring_std <= 8.0 or ring_mean > 240.0 or ring_mean < 15.0:
        return False
        
    # Считаем высокочастотную энергию шума/растра (варианты серого в 4-8px)
    blurred_full = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    hf_full = np.abs(gray.astype(np.float32) - blurred_full)
    hf_mean = float(hf_full[ring].mean())
    
    return (ring_std >= 10.0) and (hf_mean > 6.5)


def patch_density(patch: np.ndarray, thresh: int = 128) -> float:
    """Возвращает долю темных пикселей (<thresh) в патче."""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    return float((gray < thresh).mean())


def estimate_screentone_period(gray: np.ndarray, mask_boundary: np.ndarray):
    """Определяет период решетки точек скринтона Tx, Ty по автокорреляции ВЧ-слоя."""
    if not np.any(mask_boundary):
        return 1, 1
        
    gray_float = gray.astype(np.float32)
    gray_blur = cv2.GaussianBlur(gray_float, (5, 5), 0)
    hf = gray_float - gray_blur
    
    hf_ring = np.zeros_like(hf)
    hf_ring[mask_boundary] = hf[mask_boundary]
    
    autocorr_x = np.zeros(25, dtype=np.float32)
    for dx in range(1, 25):
        shifted = np.roll(hf_ring, dx, axis=1)
        valid = mask_boundary & np.roll(mask_boundary, dx, axis=1)
        if np.any(valid):
            autocorr_x[dx] = float(np.mean(hf_ring[valid] * shifted[valid]))
            
    autocorr_y = np.zeros(25, dtype=np.float32)
    for dy in range(1, 25):
        shifted = np.roll(hf_ring, dy, axis=0)
        valid = mask_boundary & np.roll(mask_boundary, dy, axis=0)
        if np.any(valid):
            autocorr_y[dy] = float(np.mean(hf_ring[valid] * shifted[valid]))
            
    tx, ty = 1, 1
    for dx in range(3, 20):
        if autocorr_x[dx] > autocorr_x[dx-1] and autocorr_x[dx] > autocorr_x[dx+1] and autocorr_x[dx] > 0.05 * (autocorr_x.max() + 1e-5):
            tx = dx
            break
            
    for dy in range(3, 20):
        if autocorr_y[dy] > autocorr_y[dy-1] and autocorr_y[dy] > autocorr_y[dy+1] and autocorr_y[dy] > 0.05 * (autocorr_y.max() + 1e-5):
            ty = dy
            break
            
    return tx, ty


def synthesize_halftone_fill(image_orig: np.ndarray, image_lama: np.ndarray, M_fail: np.ndarray, donor_forbidden: np.ndarray) -> np.ndarray:
    """
    Математический синтезатор растрового скринтона (Halftone Synthesizer).
    Генерирует 100% чистую фазовую решетку точек без шума, мыла и муара.
    """
    result = image_lama.copy()
    gray_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY) if image_orig.ndim == 3 else image_orig
    h, w = gray_orig.shape
    
    # 1. Приграничное кольцо вокруг маски M_fail
    k_block = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    block_boundary = (cv2.dilate(M_fail, k_block) > 0) & (M_fail == 0) & (~donor_forbidden)
    if not np.any(block_boundary):
        return image_lama.copy()
        
    ring_pixels = gray_orig[block_boundary]
    ring_min = float(np.percentile(ring_pixels, 10))  # Темный центр точки
    ring_max = float(np.percentile(ring_pixels, 90))  # Светлый межточечный фон
    
    # 2. Вычисляем точный период решетки Tx, Ty по автокорреляции ВЧ-слоя
    tx, ty = estimate_screentone_period(gray_orig, block_boundary)
    if tx <= 1 or ty <= 1:
        tx, ty = 6, 6
        
    # 3. Находим точную фазу точек (x0, y0) по максимальной корреляции на кольце
    gray_float = gray_orig.astype(np.float32)
    gray_blur = cv2.GaussianBlur(gray_float, (5, 5), 0)
    hf_orig = gray_float - gray_blur
    
    y_coords, x_coords = np.indices((h, w))
    
    best_phase = (0, 0)
    best_score = -float('inf')
    
    for y0 in range(ty):
        for x0 in range(tx):
            grid_x = ((x_coords - x0) % tx) - (tx / 2.0)
            grid_y = ((y_coords - y0) % ty) - (ty / 2.0)
            dist_sq = grid_x**2 + grid_y**2
            
            score = -float(np.mean(dist_sq[block_boundary] * hf_orig[block_boundary]))
            if score > best_score:
                best_score = score
                best_phase = (x0, y0)
                
    x0, y0 = best_phase
    
    # 4. Генерируем идеальный математический растровый патч
    grid_x = ((x_coords - x0) % tx) - (tx / 2.0)
    grid_y = ((y_coords - y0) % ty) - (ty / 2.0)
    dist = np.sqrt(grid_x**2 + grid_y**2)
    
    dot_ratio = float((ring_pixels < (ring_min + ring_max) / 2.0).mean())
    radius = np.sqrt(max(0.01, dot_ratio) * tx * ty / np.pi)
    
    smooth_dot = np.clip((dist - radius) / 1.2, -1.0, 1.0)
    norm_dot = (smooth_dot + 1.0) / 2.0  # [0..1]
    
    synth_gray = ring_min + norm_dot * (ring_max - ring_min)
    
    # 5. Совмещаем плавную подложку освещения LaMa (LF) и 100% чистые растровые точки (HF)
    lama_float = image_lama.astype(np.float32)
    lama_lf = cv2.GaussianBlur(lama_float, (15, 15), 0)
    
    synth_float = np.tile(synth_gray[:, :, None], (1, 1, 3)) if image_lama.ndim == 3 else synth_gray
    synth_lf = cv2.GaussianBlur(synth_float, (15, 15), 0)
    synth_hf = synth_float - synth_lf
    
    clean_halftone = np.clip(lama_lf + synth_hf, 0, 255).astype(np.uint8)
    
    # 6. Бесшовное наложение только на область растрового скринтона с плавной гауссовой альфа-маской
    gray_lama = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if result.ndim == 3 else result
    texture_pixel_mask = (M_fail > 0) & (gray_lama >= 15) & (gray_lama <= 240)
    
    alpha_mask = cv2.GaussianBlur(texture_pixel_mask.astype(np.float32), (7, 7), 2.0)
    if result.ndim == 3:
        alpha_mask = alpha_mask[:, :, np.newaxis]
        
    blended = clean_halftone.astype(np.float32) * alpha_mask + result.astype(np.float32) * (1.0 - alpha_mask)
    return np.clip(blended, 0, 255).astype(np.uint8)


def orientation_aware_donor_fill(image_orig: np.ndarray, image_lama: np.ndarray, M_fail: np.ndarray, M_text_raw: np.ndarray) -> np.ndarray:
    """
    Заполняет области M_fail с помощью математического синтезатора чистых скринтонов.
    """
    if not np.any(M_fail > 0):
        return image_lama.copy()
        
    donor_forbidden = (M_text_raw > 0)
    block_needs_texture = region_needs_texture(image_orig, M_fail, ring_width=15)
    if not block_needs_texture:
        return image_lama.copy()
        
    return synthesize_halftone_fill(image_orig, image_lama, M_fail, donor_forbidden)
