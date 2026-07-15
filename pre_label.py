import os
import cv2
import sys
import argparse
import numpy as np

# Добавляем путь для корректного импорта translator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from translator import ComicTextDetector, merge_rectangles

def pre_label_folder(img_folder):
    if not os.path.exists(img_folder):
        print(f"Ошибка: Папка {img_folder} не существует.")
        return
        
    detector = ComicTextDetector()
    if not detector.load():
        print("Ошибка: Не удалось загрузить модель ComicTextDetector.")
        return
        
    print(f"Модель ComicTextDetector успешно загружена.")
    
    # Поддерживаемые форматы изображений
    extensions = ('.png', '.jpg', '.jpeg', '.webp')
    
    # Находим все файлы изображений рекурсивно во всех подпапках
    img_files = []
    for root, dirs, filenames in os.walk(img_folder):
        for f in filenames:
            if f.lower().endswith(extensions):
                img_files.append(os.path.join(root, f))
    
    if not img_files:
        print(f"Изображения не найдены в папке: {img_folder}")
        return
        
    print(f"Найдено изображений для автоматической разметки: {len(img_files)}")
    
    # Класс 0 в YOLOv8 — text_bubble (диалоговые баблы и сюжетный текст)
    for i, img_path in enumerate(img_files):
        filename = os.path.basename(img_path)
        txt_path = os.path.splitext(img_path)[0] + '.txt'
        
        # Чтение изображения с поддержкой кириллицы в пути
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Пропуск {filename}: не удалось прочитать изображение.")
            continue
            
        h_orig, w_orig = img.shape[:2]
        
        # Запускаем детекцию (берем все кандидаты перед фильтрацией, чтобы пользователь сам убрал лишнее)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blob, scale, pad_left, pad_top = detector._preprocess(rgb)
        
        input_name = detector.session.get_inputs()[0].name
        outputs = detector.session.run(None, {input_name: blob})
        
        all_boxes = []
        all_scores = []
        
        # 1. Считываем рамки из YOLO-головы модели
        if len(outputs) >= 1 and outputs[0] is not None:
            blk = outputs[0][0]
            for row in blk:
                conf = row[4]
                if conf < 0.20:  # Понижаем порог, чтобы поймать максимум кандидатов
                    continue
                
                cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                x1 = (cx - bw / 2 - pad_left) / scale
                y1 = (cy - bh / 2 - pad_top) / scale
                x2 = (cx + bw / 2 - pad_left) / scale
                y2 = (cy + bh / 2 - pad_top) / scale
                
                x1 = max(0, min(x1, w_orig))
                y1 = max(0, min(y1, h_orig))
                x2 = max(0, min(x2, w_orig))
                y2 = max(0, min(y2, h_orig))
                
                w_rect = x2 - x1
                h_rect = y2 - y1
                
                if w_rect > 8 and h_rect > 8:
                    all_boxes.append([x1, y1, x2, y2])
                    all_scores.append(float(conf))
                    
        # Применяем NMS
        keep_indices = detector._nms(all_boxes, all_scores, iou_threshold=0.45)
        nms_boxes = [all_boxes[i] for i in keep_indices]
        
        # 2. Считываем рамки из маски сегментации для фонового текста
        seg_boxes = []
        if len(outputs) >= 3 and outputs[2] is not None:
            det_output = outputs[2]
            if len(det_output.shape) == 4 and det_output.shape[1] >= 2:
                text_mask = det_output[0, 1]
            elif len(det_output.shape) == 4 and det_output.shape[1] == 1:
                text_mask = det_output[0, 0]
            else:
                text_mask = det_output.squeeze()
                
            mask_min, mask_max = text_mask.min(), text_mask.max()
            mask_mean = text_mask.mean()
            adaptive_thresh = mask_mean + 0.6 * (mask_max - mask_mean)
            
            if mask_max - mask_min > 0.05:
                binary = (text_mask > adaptive_thresh).astype(np.uint8) * 255
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                binary = cv2.dilate(binary, kernel, iterations=2)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    x1 = (bx - pad_left) / scale
                    y1 = (by - pad_top) / scale
                    x2 = (bx + bw - pad_left) / scale
                    y2 = (by + bh - pad_top) / scale
                    
                    x1 = max(0, min(x1, w_orig))
                    y1 = max(0, min(y1, h_orig))
                    x2 = max(0, min(x2, w_orig))
                    y2 = max(0, min(y2, h_orig))
                    
                    if (x2 - x1) > 8 and (y2 - y1) > 8:
                        seg_boxes.append([x1, y1, x2, y2])
                        
        # Объединяем результаты обеих моделей
        combined = [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in nms_boxes]
        combined.extend(seg_boxes)
        merged = merge_rectangles(combined, threshold=15)
        
        # Записываем рамки в YOLO-формат (.txt файл рядом с картинкой)
        # Формат: <class_id> <x_center> <y_center> <width> <height> (значения от 0.0 до 1.0)
        with open(txt_path, 'w', encoding='utf-8') as f_out:
            for box in merged:
                x1, y1, x2, y2 = box
                w_box = x2 - x1
                h_box = y2 - y1
                
                if w_box < 8 or h_box < 8:
                    continue
                    
                x_center = (x1 + w_box / 2) / w_orig
                y_center = (y1 + h_box / 2) / h_orig
                w_norm = w_box / w_orig
                h_norm = h_box / h_orig
                
                f_out.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                
        print(f"[{i+1}/{len(img_files)}] Авто-разметка {filename} -> сохранено {len(merged)} рамок.")
        
    print("\nАвтоматическая разметка завершена!")
    print("Теперь вы можете загрузить эту папку на Roboflow (картинки вместе с созданными .txt файлами).")
    print("Сайт автоматически подгрузит все рамки, и вам останется только быстро стереть лишние звуки/баннеры!")
