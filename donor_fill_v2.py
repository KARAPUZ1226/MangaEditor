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
    1. Однородный / белая одежда (ring_std <= 6.0) -> donor НЕ нужен (оставить LaMa).
    2. Повторяющийся растровый скринтон (halftone dots, hf_mean > 7.0) -> donor ТРЕБУЕТСЯ для бесшовного растра.
    """
    k_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_width * 2 + 1, ring_width * 2 + 1))
    ring = (cv2.dilate((mask > 0).astype(np.uint8), k_ring) > 0) & (mask == 0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    ring_pixels = gray[ring]
    
    if ring_pixels.size == 0:
        return False
        
    ring_std = float(ring_pixels.std())
    if ring_std <= 6.0:
        return False
        
    # Считаем высокочастотную энергию шума/растра
    blurred_full = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    hf_full = np.abs(gray.astype(np.float32) - blurred_full)
    hf_mean = float(hf_full[ring].mean())
    
    return hf_mean > 7.0


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
    target_std_gray = float(np.std(gray_orig[block_boundary])) + 1e-5
    target_density_val = patch_density(image_orig[block_boundary], thresh=128)
    dom_angle = compute_structure_tensor_orientation(gray_orig, block_boundary)
    
    # 2. Выделяем связные компоненты провалов M_fail
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(M_fail, connectivity=8)
    
    for i in range(1, num_labels):
        x_c, y_c, w_c, h_c, area = stats[i]
        if area < 4:
            continue
            
        comp_mask = (labels == i)
        
        # 1px шаг для идеальной фазовой подгонки периодических точек скринтона!
        shifts_to_test = []
        for dy in range(-30, 31, 1):
            for dx in range(-30, 31, 1):
                if abs(dy) < 3 and abs(dx) < 3:
                    continue
                shifts_to_test.append((dy, dx))
                
        best_donor_shift = None
        best_score = float('inf')
                
        for dy, dx in shifts_to_test:
            M_shift = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted_valid = cv2.warpAffine(donor_valid_mask.astype(np.uint8), M_shift, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            if np.mean(shifted_valid[comp_mask] > 0) < 0.65:
                continue
                
            shifted_orig = cv2.warpAffine(image_orig, M_shift, (w, h), borderMode=cv2.BORDER_REFLECT)
            shifted_gray = cv2.cvtColor(shifted_orig, cv2.COLOR_BGR2GRAY)
            
            candidate_density_val = patch_density(shifted_gray[comp_mask], thresh=128)
            if abs(target_density_val - candidate_density_val) > 0.20:
                continue
                
            donor_mean_gray = float(np.mean(shifted_gray[block_boundary]))
            bright_diff = abs(donor_mean_gray - target_mean_gray)
            if bright_diff > 30.0:
                continue
                
            # Точная ошибка фазы на эталонном кольце block_boundary
            boundary_mse = float(np.mean((shifted_gray[block_boundary].astype(float) - gray_orig[block_boundary].astype(float))**2))
            donor_angle = compute_structure_tensor_orientation(shifted_gray, block_boundary)
            angle_diff = abs(np.arctan2(np.sin(dom_angle - donor_angle), np.cos(dom_angle - donor_angle)))
            
            score = boundary_mse + angle_diff * 15.0
            if score < best_score:
                best_score = score
                best_donor_shift = (dy, dx)
                
        print(f"[Donor Fill] Comp {i}: best_shift={best_donor_shift}, best_score={best_score:.2f}")
        if best_donor_shift is not None:
            dy, dx = best_donor_shift
            M_shift = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted_orig = cv2.warpAffine(image_orig, M_shift, (w, h), borderMode=cv2.BORDER_REFLECT)
            shifted_gray = cv2.cvtColor(shifted_orig, cv2.COLOR_BGR2GRAY)
            
            # Подгонка средней яркости и контраста донора под локальное кольцо (Gain & Offset)
            donor_ring_mean = float(np.mean(shifted_gray[block_boundary]))
            donor_ring_std = float(np.std(shifted_gray[block_boundary])) + 1e-5
            
            norm_donor = (shifted_orig.astype(np.float32) - donor_ring_mean) * (target_std_gray / donor_ring_std) + target_mean_gray
            donor_patch = np.clip(norm_donor, 0, 255).astype(np.uint8)
            
            result[comp_mask] = donor_patch[comp_mask]
            
    return result
