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


def smart_clean_bubbles(cv_image, bubble_items, dilation_pixels=0, lama_inpainter=None, text_segmenter=None):
    """
    Очищает и дорисовывает всю область бабла целиком.
    Будущая ИИ-модель будет передавать сюда полигоны, а пока маской выступает весь прямоугольник.
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

        if w < 1 or h < 1:
            continue

        crop = cv_image[y:y+h, x:x+w]
        if crop.size == 0:
            continue

        # Маска на 100% площади прямоугольника (или будущего полигона)
        text_mask = np.ones((h, w), dtype=np.uint8) * 255

        # Прямая дорисовка
        if lama_inpainter is not None:
            try:
                inpainted = lama_inpainter.inpaint(crop, text_mask)
                crop[:] = inpainted
            except Exception as e:
                print(f"LaMa error: {e}")
                crop[:] = cv2.inpaint(crop, text_mask, 3, cv2.INPAINT_TELEA)
        else:
            crop[:] = cv2.inpaint(crop, text_mask, 3, cv2.INPAINT_TELEA)

        cv_image[y:y+h, x:x+w] = crop
        cleaned_count += 1

    return cv_image, cleaned_count


def smart_inpaint_rect(cv_image, rect, dilation_pixels=0, lama_inpainter=None, text_segmenter=None):
    """
    Дорисовывает ручное выделение целиком.
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
    
    text_mask = np.ones((h, w), dtype=np.uint8) * 255

    if lama_inpainter is not None:
        try:
            crop = lama_inpainter.inpaint(crop, text_mask)
        except Exception as e:
            print(f"LaMa error: {e}")
            crop = cv2.inpaint(crop, text_mask, 3, cv2.INPAINT_TELEA)
    else:
        crop = cv2.inpaint(crop, text_mask, 3, cv2.INPAINT_TELEA)

    cv_image[y:y+h, x:x+w] = crop
    return cv_image
