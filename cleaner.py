import cv2
import numpy as np


from lama_mpe_pytorch import LamaMPEPyTorchInpainter as LaMaInpainter


def smart_clean_bubbles(cv_image, bubble_items, dilation_pixels=0, lama_inpainter=None, text_segmenter=None):
    if cv_image is None or not bubble_items:
        return cv_image, 0

    full_h, full_w = cv_image.shape[:2]
    cleaned_count = 0
    padding = 128  # Увеличенный контекст для LaMa для точного восстановления скринтонов

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
    padding = 128

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
