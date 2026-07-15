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


def smart_clean_bubbles(cv_image, bubble_items, dilation_pixels=5, lama_inpainter=None):
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
        # 3. Выделение кандидатов на текст через абсолютную разность от фона
        # Это позволяет одновременно выделить и светлое тело буквы, и ее темную обводку
        diff = cv2.absdiff(gray_filtered, bg_color_gray)
        
        # Динамический порог в зависимости от контрастности/шума фона
        thresh_val = max(18, int(bg_std * 2.0))
        _, text_candidates = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Дополнительно применяем локальный адаптивный порог для выделения деталей на градиентном фоне
        if bg_color_gray <= 190 and bg_std > 8.0:
            # Находим резкие перепады яркости (границы букв)
            adaptive = cv2.adaptiveThreshold(
                gray_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV if bg_color_gray > 127 else cv2.THRESH_BINARY,
                25, 3
            )
            # Объединяем оба метода для максимального охвата
            dark_mask = cv2.bitwise_or(text_candidates, adaptive)
        else:
            dark_mask = text_candidates
        # 4. Анализ связанных компонентов
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask)
        text_mask = np.zeros_like(dark_mask)
        valid_heights = []

        for i in range(1, num_labels):
            left = stats[i, cv2.CC_STAT_LEFT]
            top = stats[i, cv2.CC_STAT_TOP]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            # Вычисляем плотность объекта (отношение площади к площади его рамки)
            density = area / (width * height) if (width * height) > 0 else 0

            # Проверяем контакты с краями
            touches_left_or_right = (left <= 2 or (left + width) >= (w - 2))
            touches_top_or_bottom = (top <= 2 or (top + height) >= (h - 2))

            # Считаем объект нетекстовой границей только если:
            # 1. Он касается углов/всех краев одновременно (рамка/угол бабла)
            # 2. Он очень длинный по ширине или высоте и при этом тонкий (линия панели/бабла)
            is_border = False
            if touches_left_or_right and touches_top_or_bottom:
                is_border = True
            elif (width > w * 0.9 or height > h * 0.9) and density < 0.18:
                is_border = True

            if is_border:
                continue

            # Исключаем гигантские нетекстовые заливки (вся панель)
            if area > (w * h * 0.45) or width > w * 0.95 or height > h * 0.95:
                continue

            # Исключаем микро-шум
            if area <= 2:
                continue

            # Компонент прошел все фильтры => это текст
            text_mask[labels == i] = 255
            valid_heights.append(height)

        if np.any(text_mask == 255):
            # Вычисляем медианную высоту букв для адаптивного подбора радиуса дилатации
            median_height = np.median(valid_heights) if valid_heights else 15
            
            # Адаптивный размер расширения маски на основе высоты букв (берём 10% от высоты букв, минимум 3 пикселя)
            # Это предотвращает размытие (smudging) текстур при inpaint'е
            dilate_size = max(3, int(median_height * 0.10))
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
            dilated_mask = cv2.dilate(text_mask, kernel, iterations=1)
            
            # Защитная рамка: никогда не стираем внешние 3 пикселя кропа, 
            # чтобы дилатация случайно не задела и не размыла черные границы баблов
            dilated_mask[:3, :] = 0
            dilated_mask[-3:, :] = 0
            dilated_mask[:, :3] = 0
            dilated_mask[:, -3:] = 0
            # 6. Заполнение / Inpainting
            if bg_color_gray > 195 or bg_std < 18.0:
                # Однородный/светлый фон: заливаем сплошным цветом
                crop[dilated_mask == 255] = bg_color_bgr
            else:
                # Текстурированный фон: используем LaMa или Navier-Stokes inpaint
                if lama_inpainter is not None:
                    try:
                        inpainted_crop = lama_inpainter.inpaint(crop, dilated_mask)
                        crop[:] = inpainted_crop
                    except Exception as e:
                        print(f"LaMa inpaint error: {e}")
                        inpainted_crop = cv2.inpaint(crop, dilated_mask, 3, cv2.INPAINT_NS)
                        crop[:] = inpainted_crop
                else:
                    inpainted_crop = cv2.inpaint(crop, dilated_mask, 3, cv2.INPAINT_NS)
                    crop[:] = inpainted_crop

            cv_image[y:y+h, x:x+w] = crop
            cleaned_count += 1

    return cv_image, cleaned_count


def smart_inpaint_rect(cv_image, rect, dilation_pixels=5, lama_inpainter=None):
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
    # Определение текстовой маски на основе абсолютной разности от медианы
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bg_color_gray = int(np.median(gray))
    bg_std = np.std(gray)
    diff = cv2.absdiff(gray, bg_color_gray)
    thresh_val = max(18, int(bg_std * 2.0))
    _, thresh = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
    # Фильтруем слишком мелкие объекты
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    cleaned_thresh = np.zeros_like(thresh)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 5:
            cleaned_thresh[labels == i] = 255

    # Морфология и дилатация маски
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_thresh = cv2.morphologyEx(cleaned_thresh, cv2.MORPH_CLOSE, close_kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_pixels, dilation_pixels))
    mask_crop = cv2.dilate(cleaned_thresh, kernel, iterations=1)

    if lama_inpainter is not None:
        # Используем ИИ LaMa для Inpainting'а только на вырезанном участке (crop)
        # В LaMa лучше передавать не весь кадр, если он огромный, а только участок + паддинг,
        # но для простоты передаем crop как есть (LaMaInpainter сам ресайзит)
        inpainted_crop = lama_inpainter.inpaint(crop, mask_crop)
        inpainted = cv_image.copy()
        inpainted[y:y+h, x:x+w] = inpainted_crop
        return inpainted
    else:
        # Классический inpaint с Navier-Stokes
        full_mask = np.zeros((full_h, full_w), dtype=np.uint8)
        full_mask[y:y+h, x:x+w] = mask_crop
        inpainted = cv2.inpaint(cv_image, full_mask, 3, cv2.INPAINT_NS)
        return inpainted
