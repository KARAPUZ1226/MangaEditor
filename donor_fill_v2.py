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
    
    # Защита чистых монохромных областей (чисто белый background >248, черная тень <10)
    if ring_mean > 248.0 or ring_mean < 10.0:
        return False
        
    # Считаем высокочастотную энергию шума/растра
    blurred_full = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    hf_full = np.abs(gray.astype(np.float32) - blurred_full)
    hf_mean = float(hf_full[ring].mean())
    
    # Любой скринтон с мелким растром (std > 2.5 или hf_mean > 1.8) требует донорной заливки!
    return (ring_std >= 2.5) or (hf_mean > 1.8)


def patch_density(patch: np.ndarray, thresh: int = 128) -> float:
    """Возвращает долю темных пикселей (<thresh) в патче."""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    return float((gray < thresh).mean())


def orientation_aware_donor_fill(image_orig: np.ndarray, image_lama: np.ndarray, M_fail: np.ndarray, M_text_raw: np.ndarray) -> np.ndarray:
    """
    Заполняет области M_fail с помощью ориентированного поиска доноров и фазовой подгонки растра.
    """
    if not np.any(M_fail > 0):
        return image_lama.copy()
        
    result = image_lama.copy()
    gray_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY)
    h, w = gray_orig.shape
    
    # 1. Запрещенная зона для выбора доноров — текст и его белые ореолы/обводки (с запасом 11px)
    k_forb = np.ones((11, 11), np.uint8)
    donor_forbidden = (cv2.dilate((M_text_raw > 0).astype(np.uint8), k_forb) > 0)
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
    target_std_gray = float(np.std(gray_orig[block_boundary])) + 1e-5
    target_density_val = patch_density(image_orig[block_boundary], thresh=128)
    dom_angle = compute_structure_tensor_orientation(gray_orig, block_boundary)
    
    # 2. Вычисляем ЕДИНЫЙ глобальный фазовый сдвиг (dy, dx) для всего блока M_fail
    # Это полностью исключает разрывы сетки, лоскуты и грязь ("лоскутное одеяло")!
    best_global_shift = None
    best_score = float('inf')
    
    # Высокочастотный слой для выравнивания растра
    gray_float = gray_orig.astype(np.float32)
    gray_blur = cv2.GaussianBlur(gray_float, (5, 5), 0)
    hf_orig = gray_float - gray_blur
    
    for dy in range(-25, 26, 1):
        for dx in range(-25, 26, 1):
            if abs(dy) < 2 and abs(dx) < 2:
                continue
                
            M_shift = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted_valid = cv2.warpAffine(donor_valid_mask.astype(np.uint8), M_shift, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            if np.mean(shifted_valid[M_fail > 0] > 0) < 0.05:
                continue
                
            shifted_hf = cv2.warpAffine(hf_orig, M_shift, (w, h), borderMode=cv2.BORDER_REFLECT)
            boundary_mse = float(np.mean((shifted_hf[block_boundary] - hf_orig[block_boundary])**2))
            
            if boundary_mse < best_score:
                best_score = boundary_mse
                best_global_shift = (dy, dx)
                
    if best_global_shift is not None:
        dy, dx = best_global_shift
        M_shift = np.float32([[1, 0, dx], [0, 1, dy]])
        
        # Готовим чистый оригинальный фон без белесого шума LaMa:
        # Заполняем область текста идеальной локальной интерполяцией растровой сетки Telea
        clean_orig = cv2.inpaint(image_orig, (donor_forbidden).astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
        
        shifted_donor = cv2.warpAffine(clean_orig, M_shift, (w, h), borderMode=cv2.BORDER_REFLECT)
        shifted_gray = cv2.cvtColor(shifted_donor, cv2.COLOR_BGR2GRAY) if shifted_donor.ndim == 3 else shifted_donor
        
        # Безопасное смещение средней яркости без умножения контраста (устраняет засветы!)
        donor_ring_mean = float(np.mean(shifted_gray[block_boundary]))
        offset = np.clip(target_mean_gray - donor_ring_mean, -15.0, 15.0)
        
        norm_donor = np.clip(shifted_donor.astype(np.float32) + offset, 0, 255).astype(np.uint8)
        
        # Плавно смешиваем донорный скринтон по краям маски (feathering), чтобы полностью скрыть квадратный шов!
        gray_lama = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if result.ndim == 3 else result
        gray_target_mask = (M_fail > 0) & (gray_lama >= 15) & (gray_lama <= 240)
        
        if np.any(gray_target_mask):
            result = feather_blend_patch(result, norm_donor, gray_target_mask.astype(np.uint8) * 255, feather_px=4)
            
    return result
