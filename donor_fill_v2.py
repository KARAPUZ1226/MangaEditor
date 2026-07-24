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


def orientation_aware_donor_fill(image_orig: np.ndarray, image_lama: np.ndarray, M_fail: np.ndarray, M_text_raw: np.ndarray) -> np.ndarray:
    """
    Заполняет области M_fail с помощью фазово-выровненного донора (Phase-Locked Grid) и выборочной растровой фильтрации.
    """
    if not np.any(M_fail > 0):
        return image_lama.copy()
        
    result = image_lama.copy()
    gray_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY)
    h, w = gray_orig.shape
    
    # 1. Запрещенная зона для выбора доноров — исходные недилатированные чернила текста
    donor_forbidden = (M_text_raw > 0)
    donor_valid_mask = (~donor_forbidden) & (M_fail == 0)
    
    # Проверка "нужен ли донор для скринтона" на всем блоке M_fail
    block_needs_texture = region_needs_texture(image_orig, M_fail, ring_width=15)
    if not block_needs_texture:
        return image_lama.copy()
        
    # Кольцо 10px снаружи ВСЕГО блока M_fail для эталонного забора растровых точек
    k_block = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    block_boundary = (cv2.dilate(M_fail, k_block) > 0) & (M_fail == 0) & donor_valid_mask
    if not np.any(block_boundary):
        return image_lama.copy()
        
    target_mean_gray = float(np.mean(gray_orig[block_boundary]))
    
    # 2. Вычисляем точный период решетки скринтона Tx, Ty для идеальной фазовой состыковки
    tx, ty = estimate_screentone_period(gray_orig, block_boundary)
    
    # Формируем кандидатов сдвига strictly кратно периодам Tx, Ty (Phase-Locked Grid)
    shift_candidates = []
    if tx > 1 and ty > 1:
        for mult_y in range(-5, 6):
            for mult_x in range(-5, 6):
                if mult_y == 0 and mult_x == 0:
                    continue
                shift_candidates.append((mult_y * ty, mult_x * tx))
    else:
        for dy in range(-25, 26, 2):
            for dx in range(-25, 26, 2):
                if abs(dy) < 2 and abs(dx) < 2:
                    continue
                shift_candidates.append((dy, dx))
                
    best_global_shift = None
    best_score = float('inf')
    
    gray_float = gray_orig.astype(np.float32)
    gray_blur = cv2.GaussianBlur(gray_float, (5, 5), 0)
    hf_orig = gray_float - gray_blur
    
    for dy, dx in shift_candidates:
        M_shift = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted_valid = cv2.warpAffine(donor_valid_mask.astype(np.uint8), M_shift, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        if np.mean(shifted_valid[M_fail > 0] > 0) < 0.40:
            continue
            
        shifted_hf = cv2.warpAffine(hf_orig, M_shift, (w, h), borderMode=cv2.BORDER_REFLECT)
        boundary_mse = float(np.mean((shifted_hf[block_boundary] - hf_orig[block_boundary])**2))
        
        if boundary_mse < best_score:
            best_score = boundary_mse
            best_global_shift = (dy, dx)
            
    if best_global_shift is not None:
        dy, dx = best_global_shift
        M_shift = np.float32([[1, 0, dx], [0, 1, dy]])
        
        clean_orig = image_orig.copy()
        clean_orig[donor_forbidden] = image_lama[donor_forbidden]
        
        shifted_donor = cv2.warpAffine(clean_orig, M_shift, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Разделение на НЧ (освещение LaMa) и ВЧ (растровые точки донора)
        lama_float = image_lama.astype(np.float32)
        donor_float = shifted_donor.astype(np.float32)
        
        k_size = (15, 15)
        lama_lf = cv2.GaussianBlur(lama_float, k_size, 0)
        donor_lf = cv2.GaussianBlur(donor_float, k_size, 0)
        donor_hf = donor_float - donor_lf
        
        # Идеальный бесшовный патч: подложка освещения от LaMa + точные растровые точки от донора
        seamless_donor = np.clip(lama_lf + donor_hf, 0, 255).astype(np.uint8)
        
        # Защита черного (<=15), белого (>=240) и гладких областей (ВЧ < 4.0)
        gray_lama = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if result.ndim == 3 else result
        gray_lama_float = gray_lama.astype(np.float32)
        hf_lama = np.abs(gray_lama_float - cv2.GaussianBlur(gray_lama_float, (5, 5), 0))
        
        # Применяем бесшовный донор только на области растрового скринтона!
        texture_pixel_mask = (M_fail > 0) & (gray_lama >= 15) & (gray_lama <= 240) & (hf_lama >= 4.0)
        
        result[texture_pixel_mask] = seamless_donor[texture_pixel_mask]
        
    return result
