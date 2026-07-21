import numpy as np
import cv2


def detect_period_fft(gray, mask, sub_size=256, search_half=5):
    """
    Находит период скринтона через FFT на окрестностях вне маски.
    Возвращает (dy, dx, strength) или (0, 0, 0).
    """
    h, w = gray.shape
    half = sub_size // 2

    # Точки сетки для анализа
    grid = [
        (h // 4, w // 4), (h // 4, w // 2), (h // 4, 3 * w // 4),
        (h // 2, w // 4), (h // 2, w // 2), (h // 2, 3 * w // 4),
        (3 * h // 4, w // 4), (3 * h // 4, w // 2), (3 * h // 4, 3 * w // 4),
    ]

    best_ratio = 0.0
    best_dy, best_dx = 0, 0

    mask_u8 = (mask > 0).astype(np.uint8)

    for cy, cx in grid:
        y0 = max(0, cy - half)
        y1 = min(h, cy + half)
        x0 = max(0, cx - half)
        x1 = min(w, cx + half)
        hs, ws = y1 - y0, x1 - x0
        if hs < 64 or ws < 64:
            continue

        # Пропускаем, если в окне слишком много маски (текста) (увеличено с 0.15 до 0.65 для локальных кропов)
        text_area = np.sum(mask_u8[y0:y1, x0:x1])
        if text_area > hs * ws * 0.65:
            continue

        sub = gray[y0:y1, x0:x1].astype(np.float32)
        sub -= sub.mean()
        # Обнуляем текст внутри окна, чтобы не мешал FFT
        sub[mask_u8[y0:y1, x0:x1] > 0] = 0

        f = np.fft.fft2(sub)
        power = np.fft.fftshift(np.fft.ifft2(f * np.conj(f)).real)
        cys, cxs = hs // 2, ws // 2

        # Обнуляем центр и оси (линии рисунка)
        y_s = max(0, cys - search_half)
        y_e = min(hs, cys + search_half + 1)
        x_s = max(0, cxs - search_half)
        x_e = min(ws, cxs + search_half + 1)
        power[y_s:y_e, x_s:x_e] = 0
        power[max(0, cys - 2):min(hs, cys + 3), :] = 0
        power[:, max(0, cxs - 2):min(ws, cxs + 3)] = 0

        max_idx = np.unravel_index(np.argmax(power), power.shape)
        dy = max_idx[0] - cys
        dx = max_idx[1] - cxs
        peak = power[max_idx]
        ratio = peak / (np.mean(np.abs(power)) + 1e-5)

        if ratio > best_ratio:
            best_ratio = ratio
            best_dy = dy
            best_dx = dx

    return best_dy, best_dx, best_ratio


def periodic_fill(image, mask, period, edges_mask=None, max_r=20):
    """
    Заполняет маску периодическими сдвигами окружающей текстуры.
    Работает как ускоренный PatchMatch для скринтонов.
    Возвращает (filled_image, remaining_mask).
    """
    h, w = image.shape[:2]
    dy, dx = period
    if dy == 0 and dx == 0:
        return image.copy(), (mask > 0)

    v1 = np.array([dy, dx], dtype=np.float32)
    v2 = np.array([-dx, dy], dtype=np.float32)

    result = image.copy().astype(np.float32)
    to_fill = (mask > 0).copy()
    if edges_mask is not None:
        to_fill &= (edges_mask == 0)

    # Генерируем сдвиги по спирали
    shifts = []
    for r in range(1, max_r + 1):
        for k1 in range(-r, r + 1):
            for k2 in range(-r, r + 1):
                if abs(k1) != r and abs(k2) != r:
                    continue
                sy = int(k1 * v1[0] + k2 * v2[0])
                sx = int(k1 * v1[1] + k2 * v2[1])
                if abs(sy) < h and abs(sx) < w:
                    shifts.append((sy, sx))

    # Валидность донора: донор не в маске и не в edges
    donor_valid = (mask == 0)
    if edges_mask is not None:
        donor_valid &= (edges_mask == 0)
    donor_valid = donor_valid.astype(np.uint8) * 255

    for sy, sx in shifts:
        if not np.any(to_fill):
            break

        M = np.float32([[1, 0, sx], [0, 1, sy]])
        shifted = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        shifted_valid = cv2.warpAffine(
            donor_valid, M, (w, h),
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        can_copy = to_fill & (shifted_valid > 127)
        if np.any(can_copy):
            result[can_copy] = shifted[can_copy]
            to_fill[can_copy] = False

    return result, to_fill


def nearest_neighbor_fill(image, fill_mask):
    """
    Мгновенно заполняет оставшиеся пиксели ближайшими валидными соседями.
    fill_mask: True = нужно заполнить.
    """
    h, w = image.shape[:2]
    if not np.any(fill_mask):
        return image.copy()

    mask_inv = np.zeros((h, w), dtype=np.uint8)
    mask_inv[~fill_mask] = 255

    dist, labels = cv2.distanceTransformWithLabels(mask_inv, cv2.DIST_L2, 5)
    # labels содержит индекс ближайшего валидного пикселя (1-based, flattened)
    yy, xx = np.divmod(labels, w)

    result = image.copy().astype(np.float32)
    result[fill_mask] = image[yy[fill_mask], xx[fill_mask]]
    return result


def feather_blend(result, original, mask, kernel_size=15):
    """Смягчает границу между result и original по маске."""
    feather = cv2.GaussianBlur((mask > 0).astype(np.float32) * 255, (kernel_size, kernel_size), 0)
    feather = np.clip(feather / 255.0, 0, 1)
    if len(feather.shape) == 2 and len(original.shape) == 3:
        feather = feather[:, :, np.newaxis]
    blended = result * feather + original.astype(np.float32) * (1.0 - feather)
    return np.clip(blended, 0, 255).astype(np.uint8)


def fast_exemplar_inpaint(image, mask, edges_mask=None, screentone_threshold=3.0):
    """
    Главная функция быстрого exemplar-based инпейнтинга.
    НЕ использует нейросети, НЕ генерирует текстуру — собирает из окружающих пикселей.

    image: BGR uint8 (H, W, 3)
    mask:  uint8 (H, W), 255 = заполнить
    edges_mask: uint8 (H, W), 255 = защитить (линии рисунка)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. Ищем период скринтона
    dy, dx, strength = detect_period_fft(gray, mask)
    period_len = np.sqrt(dy**2 + dx**2)
    has_screentone = (strength > screentone_threshold) and (2.5 <= period_len <= 10.0)

    # 2. Подготовка маски
    fill_mask = (mask > 0)
    if edges_mask is not None:
        fill_mask &= (edges_mask == 0)

    # 3. Если есть скринтон — заполняем периодическими сдвигами
    if has_screentone:
        result, remaining = periodic_fill(image, mask, (dy, dx), edges_mask=edges_mask, max_r=25)
        # Если не все пиксели заполнились периодическим сдвигом, оставляем оригинальный вход (LaMa)
        # НЕ вызываем nearest_neighbor_fill, так как он создает градиентные лучи при больших масках
    else:
        result = image.copy().astype(np.float32)
        remaining = fill_mask.copy()
        if np.any(remaining):
            result = nearest_neighbor_fill(result, remaining)

    # 5. Feather blend по краям маски (уменьшено до 5 для четкости границ и предотвращения призраков)
    result = feather_blend(result, image, mask, kernel_size=5)

    # 6. Восстанавливаем структурные линии (edges) поверх всего
    if edges_mask is not None:
        # Только если edges не перекрывают текст (mask == 0)
        restore = (edges_mask > 0) & (mask == 0)
        result[restore] = image[restore]

    return result.astype(np.uint8)
