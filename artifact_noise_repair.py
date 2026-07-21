"""
artifact_noise_repair.py — Модуль Post-inpaint Artifact Repair + Noise Matching.
1. Детекция outlier-патчей (чёрные/белые кляксы)
2. Повторная точечная обработка (repair_outliers) с fallback на Telea/NS
3. Извлечение noise profile (скан/дизеринг/высокочастотный шум) из кольца оригинала
4. Наложение соответствующего шума/текстуры на inpainted регион
"""

import cv2
import numpy as np


def detect_outlier_patches(inpainted: np.ndarray, mask: np.ndarray, patch_size: int = 16, std_threshold: float = 2.5) -> np.ndarray:
    """
    Находит патчи внутри маски, которые статистически выбиваются
    из локального окружения (например, чёрная/белая клякса на светлом фоне).
    Возвращает маску патчей, требующих повторной обработки.
    """
    h, w = inpainted.shape[:2]
    gray = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY) if inpainted.ndim == 3 else inpainted
    fail_mask = np.zeros((h, w), dtype=np.uint8)

    # Кольцо вокруг маски (от 5px до 41px)
    mask_dilated_outer = cv2.dilate(mask, np.ones((41, 41), np.uint8))
    mask_dilated_inner = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    mask_ring = mask_dilated_outer & ~mask_dilated_inner
    
    ring_pixels = gray[mask_ring > 0]
    if ring_pixels.size == 0:
        return fail_mask
        
    local_mean, local_std = ring_pixels.mean(), ring_pixels.std()

    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch_mask = mask[y:y+patch_size, x:x+patch_size]
            if not np.any(patch_mask > 0):
                continue
                
            patch = gray[y:y+patch_size, x:x+patch_size]
            valid_pixels = patch[patch_mask > 0]
            if valid_pixels.size == 0:
                continue
                
            patch_mean = valid_pixels.mean()
            # Резкий выброс относительно окружения снаружи маски
            if abs(patch_mean - local_mean) > std_threshold * max(local_std, 1e-3):
                fail_mask[y:y+patch_size, x:x+patch_size] = patch_mask

    return fail_mask


def repair_outliers(image: np.ndarray, inpainted: np.ndarray, fail_mask: np.ndarray, inpaint_fn=None, max_attempts: int = 2) -> np.ndarray:
    """
    Прогоняет повторную обработку ТОЛЬКО по fail_mask с дополнительным паддингом.
    Если после max_attempts попыток outlier все еще остается — fallback на cv2.inpaint (Telea/NS).
    """
    if not np.any(fail_mask > 0):
        return inpainted.copy()

    result = inpainted.copy()
    
    for attempt in range(max_attempts):
        current_fail = detect_outlier_patches(result, fail_mask)
        if not np.any(current_fail > 0):
            break
            
        ys, xs = np.where(current_fail > 0)
        y0, y1 = max(ys.min() - 20, 0), min(ys.max() + 20, image.shape[0])
        x0, x1 = max(xs.min() - 20, 0), min(xs.max() + 20, image.shape[1])

        sub_crop = image[y0:y1, x0:x1]
        sub_mask = current_fail[y0:y1, x0:x1]
        
        if inpaint_fn is not None:
            try:
                sub_repaired = inpaint_fn(sub_crop, sub_mask)
            except Exception as e:
                print(f"[Repair Outliers] Error in donor/inpaint_fn: {e}")
                sub_repaired = cv2.inpaint(sub_crop, sub_mask, 3, cv2.INPAINT_TELEA)
        else:
            sub_repaired = cv2.inpaint(sub_crop, sub_mask, 3, cv2.INPAINT_TELEA)

        region = result[y0:y1, x0:x1]
        mask_bool = (sub_mask > 0)
        region[mask_bool] = sub_repaired[mask_bool]
        result[y0:y1, x0:x1] = region

    # Последний fallback: если все еще остались outliers — размываем через Telea
    final_fail = detect_outlier_patches(result, fail_mask)
    if np.any(final_fail > 0):
        telea_fallback = cv2.inpaint(result, final_fail, 5, cv2.INPAINT_TELEA)
        mask_bool = (final_fail > 0)
        result[mask_bool] = telea_fallback[mask_bool]

    return result


def extract_noise_profile(image: np.ndarray, mask: np.ndarray, patch_size: int = 16, ring_width: int = 20):
    """
    Рассчитывает высокочастотный шум локально по сетке патчей (16x16) из приграничного кольца.
    Возвращает карту локального std (H, W) и слой высокой частоты high_freq.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape
    
    k_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_width * 2 + 1, ring_width * 2 + 1))
    k_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ring = cv2.dilate(mask, k_outer) & ~cv2.dilate(mask, k_inner)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    high_freq = gray.astype(np.float32) - blurred.astype(np.float32)

    local_std_map = np.zeros((h, w), dtype=np.float32)
    
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            y_end = min(h, y + patch_size)
            x_end = min(w, x + patch_size)
            
            # Кольцевые пиксели вокруг текущего патча
            patch_ring = ring[max(0, y-10):min(h, y_end+10), max(0, x-10):min(w, x_end+10)]
            patch_hf = high_freq[max(0, y-10):min(h, y_end+10), max(0, x-10):min(w, x_end+10)]
            
            ring_pixels = patch_hf[patch_ring > 0]
            if ring_pixels.size > 5:
                p_std = float(ring_pixels.std())
            else:
                p_std = 0.0
                
            local_std_map[y:y_end, x:x_end] = p_std

    return high_freq, local_std_map


def apply_matched_noise(inpainted: np.ndarray, mask: np.ndarray, local_std_map: np.ndarray, high_freq: np.ndarray = None, seed=None) -> np.ndarray:
    """
    Накладывает шум локально по сетке патчей или тайлит реальный патч шума (dither/halftone).
    Временно отключается в основном пайплайне по требованию пользователя.
    """
    if isinstance(local_std_map, (int, float)):
        if local_std_map < 1e-3:
            return inpainted
        std_map = np.full(inpainted.shape[:2], local_std_map, dtype=np.float32)
    else:
        std_map = local_std_map

    if np.max(std_map) < 1e-3:
        return inpainted

    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1.0, size=inpainted.shape[:2]).astype(np.float32) * std_map

    result = inpainted.astype(np.float32).copy()
    mask_bool = (mask > 0)

    if result.ndim == 3:
        for c in range(3):
            channel = result[:, :, c]
            channel[mask_bool] += noise[mask_bool]
            result[:, :, c] = channel
    else:
        result[mask_bool] += noise[mask_bool]

    return np.clip(result, 0, 255).astype(np.uint8)
