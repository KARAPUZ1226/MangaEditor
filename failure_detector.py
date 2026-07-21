"""
failure_detector.py — Модуль детекции "провалов" LaMa (Шаг 4 Спецификации v2).
Анализирует локальную дисперсию (variance) и рассогласование гистограмм градиентов Sobel
для точечной локализации областей, где LaMa создала смаз или ошибочные штрихи.
"""

import cv2
import numpy as np


def detect_lama_failures(image_lama: np.ndarray, mask_dilated: np.ndarray, patch_size: int = 16, ring_width: int = 20) -> np.ndarray:
    """
    Возвращает бинарную маску M_fail (uint8, 255 = область провала LaMa).
    
    image_lama: BGR uint8 результат LaMa
    mask_dilated: uint8 дилатированная маска (255 = область инпейнтинга)
    """
    gray = cv2.cvtColor(image_lama, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    
    mask_bool = (mask_dilated > 0)
    if not np.any(mask_bool):
        return np.zeros((h, w), dtype=np.uint8)
        
    # 1. Формируем приграничное кольцо вокруг маски шириной ring_width px
    kernel_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_width * 2 + 1, ring_width * 2 + 1))
    outer_ring = (cv2.dilate(mask_dilated, kernel_ring) > 0) & (~mask_bool)
    
    # Расчёт эталонной дисперсии и градиентов в приграничном кольце
    if np.any(outer_ring):
        ring_pixels = gray[outer_ring]
        ring_var = float(np.var(ring_pixels))
        
        # Sobel градиенты кольца
        sobelx_ring = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely_ring = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        angles_ring = np.arctan2(sobely_ring[outer_ring], sobelx_ring[outer_ring])
        hist_ring, _ = np.histogram(angles_ring, bins=18, range=(-np.pi, np.pi), density=True)
    else:
        ring_var = 100.0
        hist_ring = np.ones(18) / 18.0
        
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    angles_full = np.arctan2(sobely, sobelx)
    magnitudes_full = np.hypot(sobelx, sobely)
    
    M_fail = np.zeros((h, w), dtype=np.uint8)
    
    # 2. Сканируем маску патчами 16x16
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            y_end = min(h, y + patch_size)
            x_end = min(w, x + patch_size)
            
            patch_mask = mask_bool[y:y_end, x:x_end]
            if not np.any(patch_mask):
                continue
                
            patch_gray = gray[y:y_end, x:x_end][patch_mask]
            if len(patch_gray) < 8:
                continue
                
            # --- Критерий 1: Потеря дисперсии (смаз) ---
            patch_var = float(np.var(patch_gray))
            var_failure = (ring_var > 15.0) and (patch_var < 0.25 * ring_var)
            
            # --- Критерий 2: Рассогласование гистограммы углов градиентов ---
            patch_angles = angles_full[y:y_end, x:x_end][patch_mask]
            patch_mags = magnitudes_full[y:y_end, x:x_end][patch_mask]
            
            # Игнорируем совсем плоские пиксели без градиента
            strong = patch_mags > 3.0
            if np.count_nonzero(strong) >= 5:
                hist_patch, _ = np.histogram(patch_angles[strong], bins=18, range=(-np.pi, np.pi), density=True)
                # Вычисление расстояния Бхаттачарии между гистограммами
                bhattacharyya = float(-np.log(np.sum(np.sqrt(hist_patch * hist_ring + 1e-8)) + 1e-8))
                angle_failure = (bhattacharyya > 0.85)
            else:
                angle_failure = (ring_var > 20.0)  # Снаружи есть четкая текстура, а внутри мыло
                
            if var_failure or angle_failure:
                M_fail[y:y_end, x:x_end][patch_mask] = 255
                
    return M_fail
