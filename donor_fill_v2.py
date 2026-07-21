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


def orientation_aware_donor_fill(image_orig: np.ndarray, image_lama: np.ndarray, M_fail: np.ndarray, M_text_raw: np.ndarray) -> np.ndarray:
    """
    Заполняет области M_fail с помощью ориентированного поиска доноров и сохранения структуры.
    
    image_orig: BGR uint8 оригинал
    image_lama: BGR uint8 результат LaMa
    M_fail: uint8 маска провалов (255 = чинить)
    M_text_raw: uint8 недилатированная маска текста (запрещенная зона для забора донора)
    """
    if not np.any(M_fail > 0):
        return image_lama.copy()
        
    result = image_lama.copy()
    gray_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY)
    gray_lama = cv2.cvtColor(image_lama, cv2.COLOR_BGR2GRAY)
    h, w = gray_orig.shape
    
    # 1. Запрещенная зона для выбора доноров — исходные недилатированные чернила текста
    donor_forbidden = (M_text_raw > 0)
    donor_valid_mask = (~donor_forbidden) & (M_fail == 0)
    
    # 2. Выделяем связные компоненты провалов M_fail
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(M_fail, connectivity=8)
    
    for i in range(1, num_labels):
        x_c, y_c, w_c, h_c, area = stats[i]
        if area < 4:
            continue
            
        comp_mask = (labels == i)
        
        # Контур компонента M_fail
        kernel_3 = np.ones((3, 3), np.uint8)
        dil = cv2.dilate(comp_mask.astype(np.uint8), kernel_3)
        boundary_mask = (dil > 0) & (~comp_mask) & donor_valid_mask
        
        # Средняя яркость и доминирующий угол на границе провала
        if np.any(boundary_mask):
            target_mean_gray = float(np.mean(gray_orig[boundary_mask]))
            dom_angle = compute_structure_tensor_orientation(gray_orig, boundary_mask)
        else:
            target_mean_gray = float(np.mean(gray_orig[comp_mask]))
            dom_angle = 0.0
            
        # Локальное окно поиска донора ±80px
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        search_radius = 80
        
        y0_s = max(0, cy - search_radius)
        y1_s = min(h, cy + search_radius)
        x0_s = max(0, cx - search_radius)
        x1_s = min(w, cx + search_radius)
        
        best_donor_shift = None
        best_score = float('inf')
        
        # Ищем идеальный вектор сдвига донора (dy, dx)
        shifts_to_test = []
        for dy in range(-40, 41, 4):
            for dx in range(-40, 41, 4):
                if abs(dy) < 4 and abs(dx) < 4:
                    continue
                shifts_to_test.append((dy, dx))
                
        for dy, dx in shifts_to_test:
            # Проверяем валидность сдвинутых пикселей
            y_shifted = np.clip(y_c + dy, 0, h - h_c)
            x_shifted = np.clip(x_c + dx, 0, w - w_c)
            
            donor_region_valid = donor_valid_mask[y0_s:y1_s, x0_s:x1_s]
            if not np.any(donor_region_valid):
                continue
                
            # Проверяем сдвинутый патч на попадание в валидного донора
            M_shift = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted_valid = cv2.warpAffine(donor_valid_mask.astype(np.uint8), M_shift, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            if not np.all(shifted_valid[comp_mask] > 0):
                continue
                
            shifted_orig = cv2.warpAffine(image_orig, M_shift, (w, h), borderMode=cv2.BORDER_REFLECT)
            shifted_gray = cv2.cvtColor(shifted_orig, cv2.COLOR_BGR2GRAY)
            
            # Считаем расхождение яркости и направления градиента
            donor_mean_gray = float(np.mean(shifted_gray[comp_mask]))
            bright_diff = abs(donor_mean_gray - target_mean_gray)
            
            if bright_diff > 25.0:
                continue
                
            # Score на совпадение градиентов
            donor_angle = compute_structure_tensor_orientation(shifted_gray, boundary_mask)
            angle_diff = abs(np.arctan2(np.sin(dom_angle - donor_angle), np.cos(dom_angle - donor_angle)))
            
            score = bright_diff + angle_diff * 15.0
            if score < best_score:
                best_score = score
                best_donor_shift = (dy, dx)
                
        # Если нашли хороший донорный патч (score < 40), вклеиваем с плавным блендингом
        if best_donor_shift is not None and best_score < 40.0:
            dy, dx = best_donor_shift
            M_shift = np.float32([[1, 0, dx], [0, 1, dy]])
            donor_patch = cv2.warpAffine(image_orig, M_shift, (w, h), borderMode=cv2.BORDER_REFLECT)
            
            # Применяем блендинг
            blended_comp = feather_blend_patch(result, donor_patch, comp_mask, feather_px=4)
            result[comp_mask] = blended_comp[comp_mask]
            
    return result
