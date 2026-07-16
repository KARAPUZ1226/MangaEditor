import cv2
import numpy as np

class LaMaInpainter:
    def __init__(self, model_path):
        import onnxruntime as ort
        # Инициализируем ONNX-сессию для запуска LaMa на процессоре (CPU)
        # Это не требует тяжелых видеокарт и работает за доли секунды
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
    def inpaint(self, img, mask):
        """
        Восстанавливает стертую область изображения с качеством один в один как в IOPaint.
        Принимает BGR-изображение и одноканальную маску (255 - стирать, 0 - сохранить).
        """
        h, w = img.shape[:2]
        
        # Препроцессинг LaMa:
        # 1. Размер кадра должен быть кратен 8 пикселям, дополняем при необходимости
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            mask_padded = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        else:
            img_padded = img
            mask_padded = mask
            
        # 2. Конвертируем в RGB и нормализуем в диапазон [0.0, 1.0]
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        img_input = img_rgb.astype(np.float32) / 255.0
        # Превращаем в формат BCHW: [1, 3, H, W]
        img_input = np.transpose(img_input, (2, 0, 1))[np.newaxis, ...]
        
        # 3. Маску переводим в формат float32 [0.0, 1.0] (1.0 - область стирания), форма [1, 1, H, W]
        mask_input = (mask_padded.astype(np.float32) > 127).astype(np.float32)
        mask_input = mask_input[np.newaxis, np.newaxis, ...]
        
        # 4. Инференс ONNX модели LaMa
        outputs = self.session.run(None, {
            'image': img_input,
            'mask': mask_input
        })
        
        # 5. Постпроцессинг результатов
        out_tensor = outputs[0][0]  # [3, H, W]
        out_img = np.transpose(out_tensor, (1, 2, 0))  # [H, W, 3]
        out_img = np.clip(out_img * 255.0, 0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        
        # Обрезаем обратно до исходного размера
        if pad_h > 0 or pad_w > 0:
            out_bgr = out_bgr[:h, :w]
            
        return out_bgr


def smart_clean_bubbles(cv_image, bubble_items, dilation_pixels=5, lama_inpainter=None, text_segmenter=None):
    """
    Очищает текст внутри всех заданных прямоугольников баблов (bubble rects).
    Анализирует локальный цвет фона бабла, подавляет шумы сканлейта (скринтоны),
    выделяет маску текста с помощью гибридной пороговой фильтрации и связанных компонентов,
    применяет продвинутые морфологические операции и восстанавливает изображение с помощью inpainting
    или адаптивной заливки в зависимости от текстурированности фона.
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

        # Clamp bounds
        x = max(0, min(x, full_w - 1))
        y = max(0, min(y, full_h - 1))
        w = min(w, full_w - x)
        h = min(h, full_h - y)

        if w < 6 or h < 6:
            continue

        crop = cv_image[y:y+h, x:x+w]
        if crop.size == 0:
            continue

        # 1. Grayscale и фильтрация
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray_filtered = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

        # 2. Оценка фона (исключаем пиксели текста для точного определения цвета и текстуры)
        bg_color_gray = int(np.median(gray_filtered))
        
        # Строим грубую предварительную маску текста, чтобы исключить его из оценки фона
        temp_thresh = int(bg_color_gray * 0.90) if bg_color_gray > 127 else int(bg_color_gray + (255 - bg_color_gray) * 0.20)
        _, temp_mask = cv2.threshold(gray_filtered, temp_thresh, 255, cv2.THRESH_BINARY_INV if bg_color_gray > 127 else cv2.THRESH_BINARY)
        
        bg_pixels_mask = (temp_mask == 0)
        if np.any(bg_pixels_mask):
            bg_color_bgr = np.median(crop[bg_pixels_mask], axis=0).astype(int).tolist()
            bg_std = np.std(gray[bg_pixels_mask])
        else:
            bg_color_bgr = np.median(crop, axis=(0, 1)).astype(int).tolist()
            bg_std = np.std(gray)
        # 3. Выделение кандидатов на текст через U-Net сегментатор или absdiff
        use_unet = False
        if text_segmenter is not None:
            try:
                # Препроцессинг под U-Net (Grayscale 256x256)
                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                crop_resized = cv2.resize(crop_gray, (256, 256))
                input_blob = crop_resized.astype(np.float32) / 255.0
                input_blob = np.expand_dims(input_blob, axis=0) # [1, 256, 256]
                input_blob = np.expand_dims(input_blob, axis=0) # [1, 1, 256, 256]
                
                # Запуск инференса
                outputs = text_segmenter.run(None, {"input": input_blob})
                logits = outputs[0][0][0]
                
                # Применяем сигмоиду
                probs = 1.0 / (1.0 + np.exp(-logits))
                mask_256 = (probs > 0.5).astype(np.uint8) * 255
                
                # Ресайзим маску обратно под исходный размер кропа
                dark_mask = cv2.resize(mask_256, (w, h), interpolation=cv2.INTER_NEAREST)
                use_unet = True
            except Exception as e:
                print(f"Ошибка сегментации U-Net: {e}. Откат на absdiff.")
                
        if not use_unet:
            # Абсолютная разность от фона (absdiff)
            diff = cv2.absdiff(gray_filtered, bg_color_gray)
            thresh_val = max(18, int(bg_std * 2.0))
            _, text_candidates = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
            
            if bg_color_gray <= 190 and bg_std > 8.0:
                adaptive = cv2.adaptiveThreshold(
                    gray_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV if bg_color_gray > 127 else cv2.THRESH_BINARY,
                    25, 3
                )
                dark_mask = cv2.bitwise_or(text_candidates, adaptive)
            else:
                dark_mask = text_candidates

        # Находим темный текст (тело букв) через простой порог яркости
        # (Манга-текст всегда черный или темно-серый, обычно < 110)
        _, dark_text = cv2.threshold(gray_filtered, 110, 255, cv2.THRESH_BINARY_INV)
        
        # Убираем одиночный микрошум
        num_labels_dt, labels_dt, stats_dt, _ = cv2.connectedComponentsWithStats(dark_text)
        cleaned_dark = np.zeros_like(dark_text)
        for i in range(1, num_labels_dt):
            if stats_dt[i, cv2.CC_STAT_AREA] > 3:
                cleaned_dark[labels_dt == i] = 255
                
        # Расширяем тело букв на 5 пикселей ( outline + 5 пикселей чтоб наверняка)
        outline_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated_dark_text = cv2.dilate(cleaned_dark, outline_kernel, iterations=1)

        # 4. Анализ связанных компонентов
        if use_unet:
            # Объединяем U-Net маску и расширенное тело букв
            text_mask = cv2.bitwise_or(dark_mask, dilated_dark_text)
        else:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask)
            text_mask = np.zeros_like(dark_mask)
            valid_heights = []

            for i in range(1, num_labels):
                left = stats[i, cv2.CC_STAT_LEFT]
                top = stats[i, cv2.CC_STAT_TOP]
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]

                density = area / (width * height) if (width * height) > 0 else 0
                touches_left_or_right = (left <= 2 or (left + width) >= (w - 2))
                touches_top_or_bottom = (top <= 2 or (top + height) >= (h - 2))

                is_border = False
                if touches_left_or_right and touches_top_or_bottom:
                    is_border = True
                elif (width > w * 0.9 or height > h * 0.9) and density < 0.18:
                    is_border = True

                if is_border:
                    continue

                if area > (w * h * 0.45) or width > w * 0.95 or height > h * 0.95:
                    continue

                if area <= 2:
                    continue

                text_mask[labels == i] = 255
                valid_heights.append(height)
                
            text_mask = cv2.bitwise_or(text_mask, dilated_dark_text)

        # Защитная рамка 8 пикселей: никогда не стираем внешние 8 пикселей кропа,
        # чтобы дилатация случайно не задела и не размыла черные границы баблов
        safety_margin = 8
        text_mask[:safety_margin, :] = 0
        text_mask[-safety_margin:, :] = 0
        text_mask[:, :safety_margin] = 0
        text_mask[:, -safety_margin:] = 0

        if np.any(text_mask == 255):
            # 6. Заполнение / Inpainting
            if bg_color_gray > 195 or bg_std < 18.0:
                # Однородный/светлый фон: заливаем сплошным цветом
                crop[text_mask == 255] = bg_color_bgr
            else:
                # Текстурированный фон: используем LaMa или Navier-Stokes inpaint
                if lama_inpainter is not None:
                    try:
                        inpainted_crop = lama_inpainter.inpaint(crop, text_mask)
                        crop[:] = inpainted_crop
                    except Exception as e:
                        print(f"Ошибка LaMa: {e}")
                        inpainted_crop = cv2.inpaint(crop, text_mask, 3, cv2.INPAINT_TELEA)
                        crop[:] = inpainted_crop
                else:
                    inpainted_crop = cv2.inpaint(crop, text_mask, 3, cv2.INPAINT_TELEA)
                    crop[:] = inpainted_crop

            cv_image[y:y+h, x:x+w] = crop
            cleaned_count += 1

    return cv_image, cleaned_count

def smart_inpaint_rect(cv_image, rect, dilation_pixels=5, lama_inpainter=None, text_segmenter=None):
    """
    Очищает фон внутри прямоугольной области на изображении.
    Если доступна модель lama_inpainter, используется она, иначе стандартный cv2.inpaint.
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
    
    use_unet = False
    if text_segmenter is not None:
        try:
            # Препроцессинг под U-Net (Grayscale 256x256)
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop_resized = cv2.resize(crop_gray, (256, 256))
            input_blob = crop_resized.astype(np.float32) / 255.0
            input_blob = np.expand_dims(input_blob, axis=0) # [1, 256, 256]
            input_blob = np.expand_dims(input_blob, axis=0) # [1, 1, 256, 256]

            # Запуск инференса
            outputs = text_segmenter.run(None, {"input": input_blob})
            logits = outputs[0][0][0]
            probs = 1.0 / (1.0 + np.exp(-logits))
            
            # Бинаризация и ресайз обратно
            mask_256 = (probs > 0.5).astype(np.uint8) * 255
            thresh = cv2.resize(mask_256, (w, h), interpolation=cv2.INTER_NEAREST)
            use_unet = True
        except Exception as e:
            print(f"Ошибка сегментации U-Net в smart_inpaint_rect: {e}")
            
    if not use_unet:
        # Абсолютная разность от фона (absdiff)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        bg_color_gray = int(np.median(gray))
        bg_std = np.std(gray)
        diff = cv2.absdiff(gray, bg_color_gray)
        thresh_val = max(18, int(bg_std * 2.0))
        _, thresh = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)

    # Находим темный текст (тело букв) через простой порог яркости
    gray_img = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, dark_text = cv2.threshold(gray_img, 110, 255, cv2.THRESH_BINARY_INV)
    
    # Убираем одиночный микрошум
    num_labels_dt, labels_dt, stats_dt, _ = cv2.connectedComponentsWithStats(dark_text)
    cleaned_dark = np.zeros_like(dark_text)
    for i in range(1, num_labels_dt):
        if stats_dt[i, cv2.CC_STAT_AREA] > 3:
            cleaned_dark[labels_dt == i] = 255
            
    # Расширяем тело букв на 5 пикселей для покрытия обводки
    outline_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_dark_text = cv2.dilate(cleaned_dark, outline_kernel, iterations=1)
    
    # Объединяем маски
    final_mask = cv2.bitwise_or(thresh, dilated_dark_text)
    
    # Защитная рамка 8 пикселей
    safety_margin = 8
    final_mask[:safety_margin, :] = 0
    final_mask[-safety_margin:, :] = 0
    final_mask[:, :safety_margin] = 0
    final_mask[:, -safety_margin:] = 0

    if np.any(final_mask == 255):
        gray_c = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        bg_color_gray_c = int(np.median(gray_c))
        bg_std_c = np.std(gray_c)
        bg_color_bgr = np.median(crop, axis=(0, 1)).astype(int).tolist()
        
        if bg_color_gray_c > 195 or bg_std_c < 18.0:
            crop[final_mask == 255] = bg_color_bgr
        else:
            if lama_inpainter is not None:
                try:
                    crop = lama_inpainter.inpaint(crop, final_mask)
                except Exception as e:
                    print(f"Ошибка LaMa inpaint в smart_inpaint_rect: {e}")
                    cv2.inpaint(crop, final_mask, 3, cv2.INPAINT_TELEA, dst=crop)
            else:
                cv2.inpaint(crop, final_mask, 3, cv2.INPAINT_TELEA, dst=crop)

    cv_image[y:y+h, x:x+w] = crop
    return cv_image
