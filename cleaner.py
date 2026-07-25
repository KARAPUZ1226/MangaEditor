import cv2
import numpy as np


from lama_mpe_pytorch import LamaMPEPyTorchInpainter as LaMaInpainter


def extract_clean_text_ink_mask(gray_crop, text_segmenter=None):
    """Единый источник истины: сегментация текста нейросетью U-Net (1:1 тайлы 256x256, порог 0.40)."""
    h, w = gray_crop.shape[:2]
    
    if text_segmenter is not None:
        try:
            inp_name = text_segmenter.get_inputs()[0].name
            
            if h <= 320 and w <= 320:
                patch_resized = cv2.resize(gray_crop, (256, 256))
                inp = (patch_resized.astype(np.float32) / 255.0)[None, None, :, :]
                outputs = text_segmenter.run(None, {inp_name: inp})
                logits = outputs[0][0, 0]
                probs_256 = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
                probs_orig = cv2.resize(probs_256, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                probs_orig = np.zeros((h, w), dtype=np.float32)
                counts_orig = np.zeros((h, w), dtype=np.float32)
                tile_size = 256
                stride = 128
                
                for y in range(0, h, stride):
                    for x in range(0, w, stride):
                        y1 = min(y, max(0, h - tile_size))
                        x1 = min(x, max(0, w - tile_size))
                        y2 = min(h, y1 + tile_size)
                        x2 = min(w, x1 + tile_size)
                        
                        patch = gray_crop[y1:y2, x1:x2]
                        if patch.shape[0] != 256 or patch.shape[1] != 256:
                            patch = cv2.resize(patch, (256, 256))
                            
                        inp_patch = (patch.astype(np.float32) / 255.0)[None, None, :, :]
                        out_patch = text_segmenter.run(None, {inp_name: inp_patch})[0][0, 0]
                        p_patch = 1.0 / (1.0 + np.exp(-np.clip(out_patch, -80.0, 80.0)))
                        
                        if (y2 - y1) != 256 or (x2 - x1) != 256:
                            p_patch = cv2.resize(p_patch, (x2 - x1, y2 - y1))
                            
                        probs_orig[y1:y2, x1:x2] += p_patch
                        counts_orig[y1:y2, x1:x2] += 1.0
                        
                probs_orig /= np.maximum(1.0, counts_orig)
                
            raw_unet_mask = (probs_orig > 0.40).astype(np.uint8) * 255
            kernel_close = np.ones((2, 2), np.uint8)
            mask_closed = cv2.morphologyEx(raw_unet_mask, cv2.MORPH_CLOSE, kernel_close)
            
            # 1. Захват белого ореола обводки (fuchidori) 5x5
            k_outline = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated = cv2.dilate(mask_closed, k_outline, iterations=1)
            white_fuchidori = (gray_crop > 200).astype(np.uint8) * 255
            mask_final = mask_closed | (dilated & white_fuchidori)
            
            # 2. Защитная окантовка 1-2px (k_safety 3x3) для полного укрытия градиента антиалиасинга
            k_safety = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            ink_mask = cv2.dilate(mask_final, k_safety, iterations=1)
            
            cv2.imwrite("DEBUG_ink_mask.png", ink_mask)
            print(f"[DEBUG_INK_MASK] sum: {ink_mask.sum()} | count_nonzero: {np.count_nonzero(ink_mask)} | shape: {ink_mask.shape}")
            return ink_mask
        except Exception as e:
            print(f"[Cleaner U-Net Error]: {e}")
            
    return np.zeros((h, w), dtype=np.uint8)


def smart_clean_bubbles(cv_image, bubble_items, dilation_pixels=0, lama_inpainter=None, text_segmenter=None):
    if cv_image is None or not bubble_items:
        return cv_image, 0

    full_h, full_w = cv_image.shape[:2]
    cleaned_count = 0
    min_padding = 96  # Гарантированный паддинг контекста со всех сторон

    seg = text_segmenter or (lama_inpainter.segmenter if (lama_inpainter and hasattr(lama_inpainter, 'segmenter')) else None)

    for bubble in bubble_items:
        rect = bubble.rect()
        pos = bubble.scenePos()
        
        # Исходные координаты
        x0 = int(pos.x() + rect.x())
        y0 = int(pos.y() + rect.y())
        w0 = int(rect.width())
        h0 = int(rect.height())

        # 1. Расширяем область до квадрата, кратного 8, не менее 512px (без растяжения!)
        S = max(w0 + 2 * min_padding, h0 + 2 * min_padding)
        S = max(S, 512)
        S = ((S + 7) // 8) * 8
        
        cx = x0 + w0 // 2
        cy = y0 + h0 // 2
        
        x = cx - S // 2
        y = cy - S // 2
        
        # Удерживаем в границах картинки
        if x < 0: x = 0
        if y < 0: y = 0
        if x + S > full_w: x = max(0, full_w - S)
        if y + S > full_h: y = max(0, full_h - S)
        
        x_end = min(full_w, x + S)
        y_end = min(full_h, y + S)

        w = x_end - x
        h = y_end - y

        if w < 1 or h < 1:
            continue

        crop = cv_image[y:y_end, x:x_end].copy()
        if crop.size == 0:
            continue

        # Маска только на текстовые символы (чернила) внутри бабла, оставляя контекст вокруг
        text_mask = np.zeros((h, w), dtype=np.uint8)
        mask_x = max(0, x0 - x)
        mask_y = max(0, y0 - y)
        mask_w = min(w - mask_x, w0)
        mask_h = min(h - mask_y, h0)
        if mask_w > 0 and mask_h > 0:
            bubble_crop = crop[mask_y:mask_y+mask_h, mask_x:mask_x+mask_w]
            bubble_gray = cv2.cvtColor(bubble_crop, cv2.COLOR_BGR2GRAY)
            text_mask[mask_y:mask_y+mask_h, mask_x:mask_x+mask_w] = extract_clean_text_ink_mask(bubble_gray, text_segmenter=seg)

        # Дорисовка
        if lama_inpainter is not None:
            try:
                lama_inpainter.full_image = cv_image
                lama_inpainter.crop_offset = (x, y)
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
    min_padding = 96

    x0 = int(rect.x())
    y0 = int(rect.y())
    w0 = int(rect.width())
    h0 = int(rect.height())

    # 1. Расширяем область до квадрата, кратного 8, не менее 512px
    S = max(w0 + 2 * min_padding, h0 + 2 * min_padding)
    S = max(S, 512)
    S = ((S + 7) // 8) * 8
    
    cx = x0 + w0 // 2
    cy = y0 + h0 // 2
    
    x = cx - S // 2
    y = cy - S // 2
    
    if x < 0: x = 0
    if y < 0: y = 0
    if x + S > full_w: x = max(0, full_w - S)
    if y + S > full_h: y = max(0, full_h - S)
    
    x_end = min(full_w, x + S)
    y_end = min(full_h, y + S)
    
    w = x_end - x
    h = y_end - y

    crop = cv_image[y:y_end, x:x_end].copy()
    
    seg = text_segmenter or (lama_inpainter.segmenter if (lama_inpainter and hasattr(lama_inpainter, 'segmenter')) else None)

    text_mask = np.zeros((h, w), dtype=np.uint8)
    mx0 = max(0, x0 - x)
    my0 = max(0, y0 - y)
    mx1 = min(w, x0 + w0 - x)
    my1 = min(h, y0 + h0 - y)
    if mx1 > mx0 and my1 > my0:
        rect_crop = crop[my0:my1, mx0:mx1]
        rect_gray = cv2.cvtColor(rect_crop, cv2.COLOR_BGR2GRAY)
        text_mask[my0:my1, mx0:mx1] = extract_clean_text_ink_mask(rect_gray)

    if dilation_pixels > 0:
        kernel = np.ones((3, 3), np.uint8)
        text_mask = cv2.dilate(text_mask, kernel, iterations=dilation_pixels)

    if lama_inpainter is not None:
        try:
            lama_inpainter.full_image = cv_image
            lama_inpainter.crop_offset = (x, y)
            crop = lama_inpainter.inpaint(crop, text_mask)
        except Exception as e:
            print(f"LaMa error: {e}")
            crop = cv2.inpaint(crop, text_mask, 3, cv2.INPAINT_TELEA)
    else:
        crop = cv2.inpaint(crop, text_mask, 3, cv2.INPAINT_TELEA)

    cv_image[y:y_end, x:x_end] = crop
    return cv_image
