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


def _build_text_mask(crop, dilation_px=2):
    """
    Строит маску текста внутри кропа бабла.
    """
    h, w = crop.shape[:2]
    if h < 6 or w < 6:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # 1. Адаптивный порог (выделение черных кандидатов на любом фоне)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    
    # 2. Находим компоненты (буквы/шум)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    
    valid_boxes = []
    char_masks = []
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cx = stats[i, cv2.CC_STAT_LEFT]
        cy = stats[i, cv2.CC_STAT_TOP]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Эвристика символов: отсекаем пыль и гигантские линии
        if 8 < area < (w * h * 0.2) and 3 < cw < w * 0.8 and 3 < ch < h * 0.8:
            valid_boxes.append((cx, cy, cw, ch))
            char_masks.append((labels == i).astype(np.uint8) * 255)
            
    if not valid_boxes:
        return None

    # 3. Кластеризация (правило близости)
    cluster_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Расширяем Bounding Box каждой буквы на 15 пикселей (зона притяжения)
    dist = 15
    for cx, cy, cw, ch in valid_boxes:
        x1 = max(0, cx - dist)
        y1 = max(0, cy - dist)
        x2 = min(w, cx + cw + dist)
        y2 = min(h, cy + ch + dist)
        cv2.rectangle(cluster_mask, (x1, y1), (x2, y2), 255, -1)
        
    c_num_labels, c_labels, _, _ = cv2.connectedComponentsWithStats(cluster_mask)
    
    cluster_counts = {i: 0 for i in range(1, c_num_labels)}
    box_cluster_ids = []
    
    for cx, cy, cw, ch in valid_boxes:
        center_x = cx + cw // 2
        center_y = cy + ch // 2
        c_id = c_labels[center_y, center_x]
        if c_id > 0:
            cluster_counts[c_id] += 1
        box_cluster_ids.append(c_id)
        
    final_text_mask = np.zeros((h, w), dtype=np.uint8)
    kept_chars = 0
    
    for i in range(len(valid_boxes)):
        c_id = box_cluster_ids[i]
        # Если в кластере больше 1 символа — это текст!
        if c_id > 0 and cluster_counts[c_id] > 1:
            final_text_mask = cv2.bitwise_or(final_text_mask, char_masks[i])
            kept_chars += 1
            
    if kept_chars == 0:
        return None
        
    # 4. Расширяем маску для захвата обводки
    if dilation_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1))
        final_text_mask = cv2.dilate(final_text_mask, kernel, iterations=1)
        
    # Защитная зона 3px
    margin = 3
    final_text_mask[:margin, :] = 0
    final_text_mask[-margin:, :] = 0
    final_text_mask[:, :margin] = 0
    final_text_mask[:, -margin:] = 0

    if not np.any(final_text_mask):
        return None

    # 5. Анализ фона (кольцо вокруг маски текста)
    bg_eval_mask = cv2.dilate(final_text_mask, np.ones((11, 11), np.uint8))
    bg_eval_mask = cv2.bitwise_xor(bg_eval_mask, final_text_mask)
    
    bg_pixels = gray[bg_eval_mask == 255]
    if len(bg_pixels) > 0:
        bg_std = float(np.std(bg_pixels))
        bg_median = int(np.median(bg_pixels))
        
        median_mask = (gray == bg_median) & (bg_eval_mask == 255)
        if np.any(median_mask):
            bg_color = np.median(crop[median_mask], axis=0).astype(int).tolist()
        else:
            bg_color = np.median(crop[bg_eval_mask == 255], axis=0).astype(int).tolist()
    else:
        bg_std = 0.0
        bg_median = 255
        bg_color = [255, 255, 255]

    return final_text_mask, bg_color, bg_median, bg_std


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

        # Твоя логика: если фон простой - заливаем белым (цветом бабла), иначе - ИИ дорисовка
        is_simple_bg = (bg_std < 12.0) or (bg_median > 240 and bg_std < 25.0)
        
        if is_simple_bg:
            crop[text_mask == 255] = bg_color
        else:
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

    is_simple_bg = (bg_std < 12.0) or (bg_median > 240 and bg_std < 25.0)
    
    if is_simple_bg:
        crop[text_mask == 255] = bg_color
    else:
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
