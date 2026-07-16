import cv2
import numpy as np


class LaMaInpainter:
    def __init__(self, model_path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    def inpaint(self, img, mask):
        """
        Восстанавливает стертую область изображения.
        Ресайзит в 512x512 (размер входа ONNX LaMa), чтобы избежать ошибок ONNX Runtime,
        и восстанавливает оригинальный размер.
        """
        h_orig, w_orig = img.shape[:2]

        # 1. Ресайз под требования ONNX-модели LaMa (512x512)
        img_resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_input = img_rgb.astype(np.float32) / 255.0
        img_input = np.transpose(img_input, (2, 0, 1))[np.newaxis, ...]

        mask_input = (mask_resized.astype(np.float32) > 127).astype(np.float32)
        mask_input = mask_input[np.newaxis, np.newaxis, ...]

        # 2. Запуск ИИ
        outputs = self.session.run(None, {'image': img_input, 'mask': mask_input})

        out_tensor = outputs[0][0]
        out_img = np.transpose(out_tensor, (1, 2, 0))
        out_img = np.clip(out_img * 255.0, 0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

        # 3. Ресайз обратно к исходным размерам кропа
        out_bgr_resized = cv2.resize(out_bgr, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)

        # 4. Переносим только стертые пиксели, чтобы не размывать оригинальный фон вокруг
        result = img.copy()
        result[mask > 127] = out_bgr_resized[mask > 127]

        return result


def smart_clean_bubbles(cv_image, bubble_items, dilation_pixels=0, lama_inpainter=None, text_segmenter=None):
    if cv_image is None or not bubble_items:
        return cv_image, 0

    full_h, full_w = cv_image.shape[:2]
    cleaned_count = 0
    padding = 64  # Контекст для LaMa

    for bubble in bubble_items:
        rect = bubble.rect()
        pos = bubble.scenePos()
        
        # Исходные координаты
        x0 = int(pos.x() + rect.x())
        y0 = int(pos.y() + rect.y())
        w0 = int(rect.width())
        h0 = int(rect.height())

        # Расширенные координаты для кропа
        x = max(0, x0 - padding)
        y = max(0, y0 - padding)
        x_end = min(full_w, x0 + w0 + padding)
        y_end = min(full_h, y0 + h0 + padding)

        w = x_end - x
        h = y_end - y

        if w < 1 or h < 1:
            continue

        crop = cv_image[y:y_end, x:x_end].copy()
        if crop.size == 0:
            continue

        # Маска только на центральную часть (сам бабл), оставляя контекст вокруг
        text_mask = np.zeros((h, w), dtype=np.uint8)
        mask_x = x0 - x
        mask_y = y0 - y
        text_mask[mask_y:mask_y+h0, mask_x:mask_x+w0] = 255

        # Дорисовка
        if lama_inpainter is not None:
            try:
                inpainted = lama_inpainter.inpaint(crop, text_mask)
                crop[:] = inpainted
            except Exception as e:
                print(f"LaMa error: {e}")
                crop[:] = cv2.inpaint(crop, text_mask, 3, cv2.INPAINT_TELEA)
        else:
            crop[:] = cv2.inpaint(crop, text_mask, 3, cv2.INPAINT_TELEA)

        cv_image[y:y_end, x:x_end] = crop
        cleaned_count += 1

    return cv_image, cleaned_count


def smart_inpaint_rect(cv_image, rect, dilation_pixels=0, lama_inpainter=None, text_segmenter=None):
    if cv_image is None or rect is None:
        return cv_image

    full_h, full_w = cv_image.shape[:2]
    padding = 64

    x0 = int(rect.x())
    y0 = int(rect.y())
    w0 = int(rect.width())
    h0 = int(rect.height())

    x = max(0, x0 - padding)
    y = max(0, y0 - padding)
    x_end = min(full_w, x0 + w0 + padding)
    y_end = min(full_h, y0 + h0 + padding)
    
    w = x_end - x
    h = y_end - y

    crop = cv_image[y:y_end, x:x_end].copy()
    
    text_mask = np.zeros((h, w), dtype=np.uint8)
    mask_x = x0 - x
    mask_y = y0 - y
    text_mask[mask_y:mask_y+h0, mask_x:mask_x+w0] = 255

    if lama_inpainter is not None:
        try:
            crop = lama_inpainter.inpaint(crop, text_mask)
        except Exception as e:
            print(f"LaMa error: {e}")
            crop = cv2.inpaint(crop, text_mask, 3, cv2.INPAINT_TELEA)
    else:
        crop = cv2.inpaint(crop, text_mask, 3, cv2.INPAINT_TELEA)

    cv_image[y:y_end, x:x_end] = crop
    return cv_image
