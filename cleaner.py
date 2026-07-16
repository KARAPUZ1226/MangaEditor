import cv2
import numpy as np


class LaMaInpainter:
    def __init__(self, model_path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    def inpaint(self, img, mask):
        """
        Восстанавливает стертую область изображения.
        Принимает BGR-изображение и одноканальную маску (255 = стирать, 0 = сохранить).
        """
        h, w = img.shape[:2]

        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8

        if pad_h > 0 or pad_w > 0:
            img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            mask_padded = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        else:
            img_padded = img
            mask_padded = mask

        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        img_input = img_rgb.astype(np.float32) / 255.0
        img_input = np.transpose(img_input, (2, 0, 1))[np.newaxis, ...]

        mask_input = (mask_padded.astype(np.float32) > 127).astype(np.float32)
        mask_input = mask_input[np.newaxis, np.newaxis, ...]

        outputs = self.session.run(None, {'image': img_input, 'mask': mask_input})

        out_tensor = outputs[0][0]
        out_img = np.transpose(out_tensor, (1, 2, 0))
        out_img = np.clip(out_img * 255.0, 0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

        if pad_h > 0 or pad_w > 0:
            out_bgr = out_bgr[:h, :w]

        return out_bgr


def _build_text_mask(crop, dilation_px=4):
    """
    Строит маску текста внутри кропа бабла.
    
    Алгоритм:
    1. Определяем медианный цвет фона (исключая предварительно найденный текст)
    2. absdiff от фона → порог → кандидаты на текст
    3. Connected components с фильтрацией шума и границ
    4. Дилатация на dilation_px пикселей для захвата обводки букв
    5. Защитная зона 3px от краёв кропа (границы баблов)
    """
    h, w = crop.shape[:2]
    if h < 6 or w < 6:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray_filtered = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # 1. Оценка фона
    bg_median = int(np.median(gray_filtered))

    # Грубая маска для исключения текста из оценки фона
    if bg_median > 127:
        temp_thresh = int(bg_median * 0.85)
        _, temp_mask = cv2.threshold(gray_filtered, temp_thresh, 255, cv2.THRESH_BINARY_INV)
    else:
        temp_thresh = int(bg_median + (255 - bg_median) * 0.25)
        _, temp_mask = cv2.threshold(gray_filtered, temp_thresh, 255, cv2.THRESH_BINARY)

    bg_pixels = (temp_mask == 0)
    if np.any(bg_pixels):
        bg_color = np.median(crop[bg_pixels], axis=0).astype(int).tolist()
        bg_std = float(np.std(gray[bg_pixels]))
    else:
        bg_color = np.median(crop, axis=(0, 1)).astype(int).tolist()
        bg_std = float(np.std(gray))

    # 2. Поиск текста: absdiff от медианы фона
    diff = cv2.absdiff(gray_filtered, bg_median)
    thresh_val = max(30, int(bg_std * 2.5))
    _, text_candidates = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)

    # 3. Connected components — фильтруем шум и границы баблов
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(text_candidates)
    text_mask = np.zeros_like(text_candidates)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cx = stats[i, cv2.CC_STAT_LEFT]
        cy = stats[i, cv2.CC_STAT_TOP]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]

        # Микрошум
        if area <= 3:
            continue

        # Гигантские заливки (вся панель или весь фон)
        if area > (w * h * 0.45) or cw > w * 0.95 or ch > h * 0.95:
            continue

        # Границы бабла: касается двух перпендикулярных краёв + тонкая линия
        density = area / (cw * ch) if (cw * ch) > 0 else 0
        touches_x = (cx <= 2 or (cx + cw) >= (w - 2))
        touches_y = (cy <= 2 or (cy + ch) >= (h - 2))

        if touches_x and touches_y:
            continue
        if (cw > w * 0.9 or ch > h * 0.9) and density < 0.18:
            continue

        text_mask[labels == i] = 255

    # 4. Дилатация — расширяем маску чтобы захватить обводку/антиалиасинг букв
    if dilation_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1))
        text_mask = cv2.dilate(text_mask, kernel, iterations=1)

    # 5. Защитная зона: не стираем ничего ближе 3px к краю кропа
    margin = 3
    text_mask[:margin, :] = 0
    text_mask[-margin:, :] = 0
    text_mask[:, :margin] = 0
    text_mask[:, -margin:] = 0

    if not np.any(text_mask):
        return None

    return text_mask, bg_color, bg_median, bg_std


def smart_clean_bubbles(cv_image, bubble_items, dilation_pixels=2, lama_inpainter=None, text_segmenter=None):
    """
    Очищает текст внутри всех заданных прямоугольников баблов.
    text_segmenter пока не используется (зарезервирован для будущей дообученной модели).
    """
    if cv_image is None or not bubble_items:
        return cv_image, 0

    full_h, full_w = cv_image.shape[:2]
    cleaned_count = 0

    for bubble in bubble_items:
        rect = bubble.rect()
        pos = bubble.scenePos()
        x = int(pos.x() + rect.x())
        y = int(pos.y() + rect.y())
        w = int(rect.width())
        h = int(rect.height())

        x = max(0, min(x, full_w - 1))
        y = max(0, min(y, full_h - 1))
        w = min(w, full_w - x)
        h = min(h, full_h - y)

        if w < 6 or h < 6:
            continue

        crop = cv_image[y:y+h, x:x+w]
        if crop.size == 0:
            continue

        result = _build_text_mask(crop, dilation_pixels)
        if result is None:
            continue

        text_mask, bg_color, bg_median, bg_std = result

        # Заполнение: LaMa дорисовывает фон, fallback — заливка медианным цветом
        if lama_inpainter is not None:
            try:
                inpainted = lama_inpainter.inpaint(crop, text_mask)
                crop[:] = inpainted
            except Exception as e:
                print(f"LaMa error: {e}")
                crop[text_mask == 255] = bg_color
        else:
            crop[text_mask == 255] = bg_color

        cv_image[y:y+h, x:x+w] = crop
        cleaned_count += 1

    return cv_image, cleaned_count


def smart_inpaint_rect(cv_image, rect, dilation_pixels=2, lama_inpainter=None, text_segmenter=None):
    """
    Очищает текст внутри ручного прямоугольного выделения.
    """
    if cv_image is None or rect is None:
        return cv_image

    x = int(rect.x())
    y = int(rect.y())
    w = int(rect.width())
    h = int(rect.height())

    full_h, full_w = cv_image.shape[:2]
    x = max(0, min(x, full_w - 1))
    y = max(0, min(y, full_h - 1))
    w = max(1, min(w, full_w - x))
    h = max(1, min(h, full_h - y))

    crop = cv_image[y:y+h, x:x+w].copy()

    result = _build_text_mask(crop, dilation_pixels)
    if result is None:
        return cv_image

    text_mask, bg_color, bg_median, bg_std = result

    if lama_inpainter is not None:
        try:
            crop = lama_inpainter.inpaint(crop, text_mask)
        except Exception as e:
            print(f"LaMa error: {e}")
            crop[text_mask == 255] = bg_color
    else:
        crop[text_mask == 255] = bg_color

    cv_image[y:y+h, x:x+w] = crop
    return cv_image
