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
    Использует эвристику "Ореола" (Halo) для отделения текста от спидлайнов и скринтонов.
    """
    h, w = crop.shape[:2]
    if h < 6 or w < 6: return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # 1. Ищем всё тёмное (потенциальный текст и линии)
    # Адаптируем порог под общую яркость кропа
    crop_median = int(np.median(gray))
    thresh_val = min(140, max(60, crop_median - 40)) 
    _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    valid_chars = []
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Отсекаем пыль и рамки панелей
        if 10 < area < (w * h * 0.4) and 3 < cw < w * 0.9 and 3 < ch < h * 0.9:
            char_mask = (labels == i).astype(np.uint8) * 255
            
            # 2. ЭВРИСТИКА ОБВОДКИ (Halo Rule)
            # Расширяем букву на 3 пикселя, чтобы получить кольцо вокруг неё
            dilated = cv2.dilate(char_mask, np.ones((5,5), np.uint8))
            halo = cv2.bitwise_xor(dilated, char_mask)
            
            # Вычитаем другие тёмные объекты, чтобы соседние буквы не портили статистику
            halo_clean = cv2.bitwise_and(halo, cv2.bitwise_not(binary))
            
            halo_pixels = gray[halo_clean == 255]
            if len(halo_pixels) > 0:
                halo_median = int(np.median(halo_pixels))
                # Если вокруг буквы светло (белый фон или белая обводка) — это текст!
                # Если вокруг серо (скринтон) или темно (рисунок) — это мусор (спидлайны)
                if halo_median > 175:
                    valid_chars.append(char_mask)

    if not valid_chars:
        return None

    # Объединяем подтверждённые буквы
    text_mask = np.zeros((h, w), dtype=np.uint8)
    for mask in valid_chars:
        text_mask = cv2.bitwise_or(text_mask, mask)
        
    # 3. Дилатация (чтобы стереть саму белую обводку тоже)
    if dilation_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1))
        text_mask = cv2.dilate(text_mask, kernel, iterations=1)
        
    # Защитная зона 3px
    margin = 3
    text_mask[:margin, :] = 0
    text_mask[-margin:, :] = 0
    text_mask[:, :margin] = 0
    text_mask[:, -margin:] = 0

    if not np.any(text_mask):
        return None

    # 4. Анализ фона (для выбора между заливкой и LaMa)
    bg_eval_mask = cv2.dilate(text_mask, np.ones((11, 11), np.uint8))
    bg_eval_mask = cv2.bitwise_xor(bg_eval_mask, text_mask)
    
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
