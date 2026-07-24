import cv2
import numpy as np


def region_needs_texture(image_orig: np.ndarray, M_fail: np.ndarray, ring_width: int = 15) -> bool:
    """
    Проверяет, окружена ли область маски M_fail растровым скринтоном.
    """
    gray = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY) if image_orig.ndim == 3 else image_orig
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_width * 2 + 1, ring_width * 2 + 1))
    boundary = (cv2.dilate(M_fail, k) > 0) & (M_fail == 0)
    
    if not np.any(boundary):
        return False
        
    ring_pixels = gray[boundary].astype(np.float32)
    blur_pixels = cv2.GaussianBlur(gray, (5, 5), 0)[boundary].astype(np.float32)
    hf_diff = np.abs(ring_pixels - blur_pixels)
    
    return float(np.mean(hf_diff)) > 3.0


def orientation_aware_donor_fill(image_orig: np.ndarray, image_lama: np.ndarray, M_fail: np.ndarray, M_text_raw: np.ndarray) -> np.ndarray:
    """
    Заполняет области M_fail с помощью 100% точного фазово-выровненного оригинального растра (Direct Ring Phase Lock).
    """
    if not np.any(M_fail > 0):
        return image_lama.copy()
        
    gray_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY) if image_orig.ndim == 3 else image_orig
    h, w = gray_orig.shape
    
    donor_forbidden = (M_text_raw > 0)
    
    # Проверка "нужен ли донор для скринтона"
    block_needs_texture = region_needs_texture(image_orig, M_fail, ring_width=15)
    if not block_needs_texture:
        return image_lama.copy()
        
    # Кольцо 15px снаружи M_fail для нахождения фазы
    k_block = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    block_boundary = (cv2.dilate(M_fail, k_block) > 0) & (M_fail == 0) & (~donor_forbidden)
    if not np.any(block_boundary):
        return image_lama.copy()
        
    valid_donor_zone = (~donor_forbidden)
    
    best_shift = (0, 0)
    best_err = float('inf')
    
    # Сканируем фазовые сдвиги (dy, dx) от -40px до +40px
    for dy in range(-40, 41):
        for dx in range(-40, 41):
            if abs(dy) < 2 and abs(dx) < 2:
                continue
                
            M_shift = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted_valid = cv2.warpAffine(valid_donor_zone.astype(np.uint8), M_shift, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
            
            if np.count_nonzero(shifted_valid & (M_fail > 0)) < 0.30 * np.count_nonzero(M_fail > 0):
                continue
                
            shifted_orig = cv2.warpAffine(gray_orig, M_shift, (w, h), borderMode=cv2.BORDER_REFLECT)
            ring_valid = block_boundary & shifted_valid
            if not np.any(ring_valid):
                continue
                
            l1_err = float(np.mean(np.abs(shifted_orig[ring_valid].astype(np.float32) - gray_orig[ring_valid].astype(np.float32))))
            
            if l1_err < best_err:
                best_err = l1_err
                best_shift = (dy, dx)
                
    dy, dx = best_shift
    if best_err > 45.0 or (dy == 0 and dx == 0):
        return image_lama.copy()
        
    M_shift = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted_donor = cv2.warpAffine(image_orig, M_shift, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # Разделение на НЧ (освещение LaMa) и ВЧ (100% реальные точки оригинального скринтона)
    lama_float = image_lama.astype(np.float32)
    donor_float = shifted_donor.astype(np.float32)
    
    k_size = (15, 15)
    lama_lf = cv2.GaussianBlur(lama_float, k_size, 0)
    donor_lf = cv2.GaussianBlur(donor_float, k_size, 0)
    donor_hf = donor_float - donor_lf
    
    # Замена НЧ на LaMa (0% швов), сохраняя 100% оригинальные растровые точки (HF)
    clean_halftone = np.clip(lama_lf + donor_hf, 0, 255).astype(np.uint8)
    
    gray_lama = cv2.cvtColor(image_lama, cv2.COLOR_BGR2GRAY) if image_lama.ndim == 3 else image_lama
    texture_pixel_mask = (M_fail > 0) & (gray_lama >= 15) & (gray_lama <= 240)
    
    alpha_mask = cv2.GaussianBlur(texture_pixel_mask.astype(np.float32), (7, 7), 2.0)
    if image_lama.ndim == 3:
        alpha_mask = alpha_mask[:, :, np.newaxis]
        
    blended = clean_halftone.astype(np.float32) * alpha_mask + image_lama.astype(np.float32) * (1.0 - alpha_mask)
    return np.clip(blended, 0, 255).astype(np.uint8)
