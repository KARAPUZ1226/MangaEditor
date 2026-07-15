import cv2
import requests
import math
import os
import numpy as np
from PySide6.QtCore import QRectF

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

def translate_google(text, src_lang='auto', dest_lang='ru'):
    if not text.strip():
        return ""
    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={src_lang}&tl={dest_lang}&dt=t&q={requests.utils.quote(text)}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            result = response.json()
            translated = "".join([part[0] for part in result[0] if part[0]])
            return translated
    except Exception as e:
        print(f"Translation API error: {e}")
    return "[Ошибка автоперевода]"

def merge_rectangles(rects, threshold=20):
    """Merges overlapping or closely located rectangles."""
    if not rects:
        return []
    to_merge = [list(r) for r in rects]
    merged = []
    while to_merge:
        current = to_merge.pop(0)
        has_merged = True
        while has_merged:
            has_merged = False
            for i in range(len(to_merge) - 1, -1, -1):
                candidate = to_merge[i]
                if not (current[2] + threshold < candidate[0] or candidate[2] + threshold < current[0] or
                        current[3] + threshold < candidate[1] or candidate[3] + threshold < current[1]):
                    current = [
                        min(current[0], candidate[0]),
                        min(current[1], candidate[1]),
                        max(current[2], candidate[2]),
                        max(current[3], candidate[3])
                    ]
                    to_merge.pop(i)
                    has_merged = True
        merged.append(current)
    return merged


# ===========================================================================
# ComicTextDetector — специализированная нейросеть для манги
# ===========================================================================
class ComicTextDetector:
    """
    Обнаруживает текстовые регионы (баблы и надписи) на страницах манги/комиксов
    с помощью ONNX-модели comictextdetector (YOLOv5-backbone + UNet-сегментация).

    Модель обучена именно на manga/comic данных, поэтому она:
    - Безошибочно находит японский вертикальный текст
    - Игнорирует лица, рисунки, текстуры
    - Работает с баблами любой формы
    """
    MODEL_INPUT_SIZE = 1024  # Модель обучена на 1024x1024

    def __init__(self, model_path=None):
        self.session = None
        if model_path is None:
            custom_path = os.path.join(os.path.dirname(__file__), "models", "custom_detector.onnx")
            if os.path.exists(custom_path):
                model_path = custom_path
                self.model_input_size = 640
                self.is_custom_model = True
                print(f"[Детектор] Загружаем вашу собственную обученную модель: {model_path}")
            else:
                model_path = os.path.join(os.path.dirname(__file__), "models", "comictextdetector.pt.onnx")
                self.model_input_size = 1024
                self.is_custom_model = False
        else:
            self.model_path = model_path
            self.is_custom_model = "custom_detector" in model_path
            self.model_input_size = 640 if self.is_custom_model else 1024
        self.model_path = model_path

    def load(self):
        if self.session is not None:
            return True
        if not ONNX_AVAILABLE:
            print("onnxruntime не установлен!")
            return False
        if not os.path.exists(self.model_path):
            print(f"Модель не найдена: {self.model_path}")
            return False
        try:
            self.session = onnxruntime.InferenceSession(
                self.model_path,
                providers=['CPUExecutionProvider']
            )
            return True
        except Exception as e:
            print(f"Ошибка загрузки ONNX модели: {e}")
            return False

    def _preprocess(self, cv_image):
        """Подготовка изображения для входа модели: resize + нормализация."""
        h, w = cv_image.shape[:2]
        # letterbox resize к model_input_size x model_input_size
        scale = min(self.model_input_size / h, self.model_input_size / w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(cv_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Создаем квадратный холст (черный letterbox)
        canvas = np.zeros((self.model_input_size, self.model_input_size, 3), dtype=np.uint8)
        pad_top = (self.model_input_size - new_h) // 2
        pad_left = (self.model_input_size - new_w) // 2
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

        # Нормализация [0, 1], HWC -> CHW, batch dim
        blob = canvas.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))  # CHW
        blob = np.expand_dims(blob, 0)  # NCHW
        return blob, scale, pad_left, pad_top

    def _nms(self, boxes, scores, iou_threshold=0.45):
        """Non-Maximum Suppression для удаления дублирующихся рамок."""
        if len(boxes) == 0:
            return []
        boxes_arr = np.array(boxes, dtype=np.float32)
        scores_arr = np.array(scores, dtype=np.float32)
        
        x1 = boxes_arr[:, 0]
        y1 = boxes_arr[:, 1]
        x2 = boxes_arr[:, 2]
        y2 = boxes_arr[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        order = scores_arr.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w_inter = np.maximum(0, xx2 - xx1)
            h_inter = np.maximum(0, yy2 - yy1)
            intersection = w_inter * h_inter
            
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / (union + 1e-6)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep

    def detect(self, cv_image):
        """
        Запускает детектирование текстовых регионов на изображении.
        
        Модель comictextdetector выдает 3 выхода:
          outputs[0] = blk — bounding boxes [1, N, 7] (cx, cy, w, h, conf, cls1, cls2)
          outputs[1] = seg — общая маска сегментации [1, 1, 1024, 1024]
          outputs[2] = det — двухканальная маска детекции [1, 2, 1024, 1024]
        
        Мы используем bounding boxes (output[0]) с NMS как основной метод,
        и адаптивную маску (output[2]) как дополнение.
        """
        if not self.load():
            return []

        h_orig, w_orig = cv_image.shape[:2]
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        blob, scale, pad_left, pad_top = self._preprocess(rgb)

        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: blob})

        all_boxes = []
        all_scores = []

        # === 1. Bounding boxes из output[0] (blk) — основной метод ===
        if len(outputs) >= 1 and outputs[0] is not None:
            blk = outputs[0]
            if len(blk.shape) == 3:
                blk = blk[0]
            if self.is_custom_model:
                # В YOLOv8 вывод имеет форму [5, 8400] -> транспонируем в [8400, 5]
                blk = blk.T
            if len(blk.shape) == 2 and blk.shape[1] >= 5:
                conf_threshold = 0.35
                for row in blk:
                    conf = row[4]
                    if conf < conf_threshold:
                        continue
                        
                    # Отсеиваем звуковые эффекты (SFX) и декор
                    # Класс 0 (row[5]) - звуки / непереводимое
                    # Класс 1 (row[6]) - обычный текст
                    if len(row) >= 7:
                        cls1, cls2 = row[5], row[6]
                        if cls1 > cls2:
                            continue # Игнорируем SFX!

                    cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                    
                    # Конвертируем из координат модели в оригинальные координаты
                    x1 = (cx - bw / 2 - pad_left) / scale
                    y1 = (cy - bh / 2 - pad_top) / scale
                    x2 = (cx + bw / 2 - pad_left) / scale
                    y2 = (cy + bh / 2 - pad_top) / scale

                    # Клипим по границам
                    x1 = max(0, min(x1, w_orig))
                    y1 = max(0, min(y1, h_orig))
                    x2 = max(0, min(x2, w_orig))
                    y2 = max(0, min(y2, h_orig))

                    w_rect = x2 - x1
                    h_rect = y2 - y1

                    if w_rect > 4 and h_rect > 4:
                        if w_rect < w_orig * 0.85 and h_rect < h_orig * 0.85:
                            all_boxes.append([x1, y1, x2, y2])
                            all_scores.append(float(conf))

        # === 2. NMS для удаления дублей ===
        if all_boxes:
            keep_indices = self._nms(all_boxes, all_scores, iou_threshold=0.4)
            nms_boxes = [all_boxes[i] for i in keep_indices]
        else:
            nms_boxes = []

        # === 3. Дополнительно: маска сегментации из output[2] (det) ===
        seg_boxes = []
        if len(outputs) >= 3 and outputs[2] is not None:
            det_output = outputs[2]
            # Форма: [1, 2, H, W] — канал 1 = текст
            if len(det_output.shape) == 4 and det_output.shape[1] >= 2:
                text_mask = det_output[0, 1]
            elif len(det_output.shape) == 4 and det_output.shape[1] == 1:
                text_mask = det_output[0, 0]
            else:
                text_mask = det_output.squeeze()

            # Адаптивная бинаризация: порог = среднее + 0.6 * (макс - среднее)
            mask_min, mask_max = text_mask.min(), text_mask.max()
            mask_mean = text_mask.mean()
            adaptive_thresh = mask_mean + 0.6 * (mask_max - mask_mean)
            
            if mask_max - mask_min > 0.05:  # Есть разброс => есть текст
                binary = (text_mask > adaptive_thresh).astype(np.uint8) * 255

                # Морфология
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                binary = cv2.dilate(binary, kernel, iterations=2)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                           cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))

                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    if bw < 4 or bh < 4:
                        continue
                    x1 = (bx - pad_left) / scale
                    y1 = (by - pad_top) / scale
                    x2 = (bx + bw - pad_left) / scale
                    y2 = (by + bh - pad_top) / scale
                    x1 = max(0, min(x1, w_orig))
                    y1 = max(0, min(y1, h_orig))
                    x2 = max(0, min(x2, w_orig))
                    y2 = max(0, min(y2, h_orig))
                    w_rect = x2 - x1
                    h_rect = y2 - y1
                    if w_rect > 4 and h_rect > 4:
                        if w_rect < w_orig * 0.7 and h_rect < h_orig * 0.7:
                            seg_boxes.append([int(x1), int(y1), int(x2), int(y2)])

        # === 4. Объединяем bbox и seg результаты ===
        combined = [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in nms_boxes]
        # Восстанавливаем seg_boxes, так как контекстный OCR-фильтр теперь сам отсеет звуки
        combined.extend(seg_boxes)
        merged = merge_rectangles(combined, threshold=6)

        # === 5. Конвертируем в QRectF с padding и фильтрацией ===
        result = []
        for box in merged:
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            
            # Фильтр по минимальному размеру (делаем мягче, чтобы ловить узкие баблы типа 21px)
            if w < 4 or h < 4:
                continue
            
            # Фильтр по соотношению сторон (делаем мягче для длинных вертикальных колонок текста)
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > 15:
                continue
            
            # Фильтр: если рамка занимает более 60% площади страницы — скорее всего ошибка
            if (w * h) > (w_orig * h_orig * 0.6):
                continue

            # Padding: немного расширяем рамку чтобы захватить краюшки текста
            pad = max(4, int(min(w, h) * 0.08))
            qr = QRectF(
                max(0, x1 - pad),
                max(0, y1 - pad),
                min(w + 2 * pad, w_orig - max(0, x1 - pad)),
                min(h + 2 * pad, h_orig - max(0, y1 - pad))
            )
            result.append(qr)

        return result


# ===========================================================================
# ИИ Очистка баблов (LaMa Inpainting)
# ===========================================================================
class LamaInpainter:
    """Обертка над ONNX моделью LaMa (Large Mask Inpainting) для удаления текста с фона"""
    def __init__(self, model_path='models/lama-manga.onnx'):
        self.model_path = model_path
        self.session = None

    def load(self):
        if self.session is not None:
            return True
        if not os.path.exists(self.model_path):
            print(f"[LaMa] Модель не найдена: {self.model_path}")
            return False
        try:
            self.session = onnxruntime.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
            print("[LaMa] ONNX модель загружена.")
            return True
        except Exception as e:
            print(f"[LaMa] Ошибка загрузки ONNX: {e}")
            return False

    def inpaint(self, image, mask):
        """
        image: np.array BGR (H, W, 3)
        mask: np.array (H, W) or (H, W, 1), 255 - текст для удаления
        """
        if self.session is None:
            if not self.load():
                return image

        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, 2)
            
        h, w = image.shape[:2]
        size = max(h, w)
        
        # Pad to square (to keep aspect ratio during resize)
        padded_img = np.zeros((size, size, 3), dtype=np.uint8)
        padded_mask = np.zeros((size, size, 1), dtype=np.uint8)
        padded_img[:h, :w] = image
        padded_mask[:h, :w] = mask
        
        # Resize to 512x512
        img_512 = cv2.resize(padded_img, (512, 512))
        mask_512 = cv2.resize(padded_mask, (512, 512))
        
        # Prepare inputs
        img_tensor = np.transpose(img_512.astype(np.float32) / 255.0, (2, 0, 1))
        img_tensor = np.expand_dims(img_tensor, 0)
        mask_tensor = np.expand_dims(mask_512.astype(np.float32) / 255.0, 0)
        if len(mask_tensor.shape) == 3: 
            mask_tensor = np.expand_dims(mask_tensor, 0)
        
        # Run inference
        try:
            out = self.session.run(None, {'image': img_tensor, 'mask': mask_tensor})[0]
            
            # Postprocess
            out_img = out[0] # 3, 512, 512
            out_img = np.transpose(out_img, (1, 2, 0)) * 255.0
            out_img = np.clip(out_img, 0, 255).astype(np.uint8)
            
            # Resize back and crop
            out_resized = cv2.resize(out_img, (size, size))
            return out_resized[:h, :w]
        except Exception as e:
            print(f"[LaMa] Ошибка инференса: {e}")
            return image
class EasyOCRManager:
    def __init__(self):
        self.detector_reader = None
        self.readers = {}
        self.comic_detector = ComicTextDetector()
        self.lama_inpainter = LamaInpainter()
        self.manga_ocr = None

    def get_manga_ocr(self):
        if self.manga_ocr is None:
            try:
                from manga_ocr import MangaOcr
                print("[MangaOCR] Инициализация offline-модели японского OCR...")
                self.manga_ocr = MangaOcr()
                print("[MangaOCR] Модель успешно загружена!")
            except Exception as e:
                print(f"[MangaOCR] Не удалось загрузить офлайн-модель: {e}")
                self.manga_ocr = False
        return self.manga_ocr if self.manga_ocr is not False else None

    def _is_unwanted_text(self, text, lang):
        if not text.strip():
            return False
            
        text = text.lower().strip()
        
        # 1. Фильтр баннеров и авторских кредитов
        credit_keywords = [
            "原作", "漫画", "キャラクター", "原案", "構成", "作画", "翻訳",
            "copyright", "©", "ch.", "chapter", "vol.", "page", "http", "www", ".com",
            "twitter", "pixiv", "skeb", "fanbox", "illust", "art"
        ]
        for keyword in credit_keywords:
            if keyword in text:
                print(f"[Контекст-Фильтр] Отсеяли баннер/кредиты: '{text}'")
                return True
                
        # 2. Фильтр звуковых эффектов (SFX) для японского языка
        if lang == 'ja':
            import re
            
            # Очистим текст от знаков препинания для анализа (используем a-zA-Z0-9 вместо \w, чтобы не удалять кандзи/хирагану)
            clean_text = re.sub(r'[\s0-9a-zA-Z\?\!\.\,\-\_\#\$\%\&\*\(\)\+\[\]\{\}\:\;\"\'\<\>\~\=\|\\/`・…―ー■．／＼％＆＊＃＠？！＋＝]', '', text)
            clean_len = len(clean_text)
            
            if clean_len == 0:
                return False
                
            # Проверяем наличие хираганы, катаканы и кандзи
            has_hiragana = bool(re.search(r'[\u3040-\u309f]', clean_text))
            has_katakana = bool(re.search(r'[\u30a0-\u30ff]', clean_text))
            has_kanji = bool(re.search(r'[\u4e00-\u9faf]', clean_text))
            
            # Проверяем на повторяющиеся звуки (SFX) типа "ゴゴゴ", "ドキドキ", "だらだら"
            is_repeated_sfx = False
            if clean_len >= 2:
                if len(set(clean_text)) == 1:
                    is_repeated_sfx = True
                elif clean_len >= 4 and clean_len % 2 == 0:
                    half = clean_len // 2
                    if clean_text[:half] == clean_text[half:]:
                        is_repeated_sfx = True
                        
            if is_repeated_sfx:
                print(f"[Контекст-Фильтр] Отсеяли повторяющийся звук (SFX): '{text}'")
                return True
                
            # Если это одиночный символ (кроме кандзи или знаков вопроса/восклицания)
            if clean_len <= 1 and not has_kanji:
                if not ('?' in text or '？' in text or '!' in text or '！' in text):
                    print(f"[Контекст-Фильтр] Отсеяли одиночный символ/шум: '{text}'")
                    return True
                    
            # Является ли это связным японским текстом?
            # Связный текст должен содержать либо кандзи, либо японские грамматические частицы (の, は, が, を, に, で, と, て, た, だ, ね, よ, か, も)
            # либо быть известным коротким словом диалога
            common_dialogue_words = ["はい", "いいえ", "うん", "うーん", "あ", "え", "お", "おい", "ねえ", "さあ", "ありがとう", "ごめん"]
            
            # Проверяем наличие частиц (но не в начале слова, так как это может быть началом SFX)
            has_particles = False
            if clean_len >= 2:
                has_particles = bool(re.search(r'.[のはがをにでとてただしねよかも]', clean_text))
                
            # Проверяем, является ли это заимствованным словом (катакана) >= 3 символов (типа ベッド, マスター)
            is_loan_word = has_katakana and clean_len >= 3
            
            is_coherent = has_hiragana or has_kanji or has_particles or is_loan_word or (clean_text in common_dialogue_words)
            
            if not is_coherent:
                print(f"[Контекст-Фильтр] Отсеяли несвязный текст (вероятный звук/SFX): '{text}'")
                return True
                
        return False

    def get_detector_reader(self):
        if self.detector_reader is None and EASYOCR_AVAILABLE:
            self.detector_reader = easyocr.Reader(['en'])
        return self.detector_reader

    def get_transcribe_reader(self, lang):
        if not EASYOCR_AVAILABLE:
            return None
        if lang not in self.readers:
            try:
                if lang == 'ja':
                    self.readers[lang] = easyocr.Reader(['ja', 'en'])
                elif lang == 'ko':
                    self.readers[lang] = easyocr.Reader(['ko', 'en'])
                else:
                    self.readers[lang] = easyocr.Reader(['en'])
            except Exception as e:
                print(f"Error loading reader for {lang}: {e}")
                self.readers[lang] = self.detector_reader
        return self.readers[lang]

    def detect_bubbles_ai(self, cv_image, lang='ja'):
        """
        Обнаруживает текстовые баблы с помощью специализированной ONNX-модели
        ComicTextDetector, с последующей умной OCR фильтрацией шума/звуков.
        """
        if ONNX_AVAILABLE:
            results = self.comic_detector.detect(cv_image)
            if results:
                # Фильтруем баблы по контексту (OCR + правила)
                filtered_results = []
                for qr in results:
                    x1, y1 = int(qr.x()), int(qr.y())
                    w, h = int(qr.width()), int(qr.height())
                    h_orig, w_orig = cv_image.shape[:2]
                    
                    # Ограничиваем координаты
                    x1 = max(0, min(x1, w_orig - 1))
                    y1 = max(0, min(y1, h_orig - 1))
                    w = min(w, w_orig - x1)
                    h = min(h, h_orig - y1)
                    
                    if w < 4 or h < 4:
                        continue
                        
                    crop = cv_image[y1:y1+h, x1:x1+w]
                    
                    # Распознаем текст внутри бабла для анализа контекста
                    text = ""
                    if lang == 'ja':
                        mocr = self.get_manga_ocr()
                        if mocr:
                            try:
                                from PIL import Image
                                pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                                text = mocr(pil_img)
                            except Exception as e:
                                print(f"Manga-OCR error during filtering: {e}")
                        
                        # Если manga-ocr не сработал, fallback на EasyOCR
                        if not text.strip():
                            reader = self.get_transcribe_reader(lang)
                            if reader:
                                try:
                                    res = reader.readtext(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                                    text = " ".join([r[1] for r in res])
                                except Exception as e:
                                    print(f"EasyOCR error during filtering fallback: {e}")
                    else:
                        reader = self.get_transcribe_reader(lang)
                        if reader:
                            try:
                                res = reader.readtext(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                                text = " ".join([r[1] for r in res])
                            except Exception as e:
                                print(f"EasyOCR error during filtering: {e}")
                                
                    # Проверяем, является ли текст нежелательным (звуки, кредиты)
                    if not self._is_unwanted_text(text, lang):
                        filtered_results.append(qr)
                        
                return filtered_results

        # Fallback: CRAFT (EasyOCR) + OpenCV контуры
        reader = self.get_transcribe_reader(lang)
        if not reader:
            return self._fallback_contour_detect(cv_image)

        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        try:
            horiz_boxes, free_boxes = reader.detect(rgb)
        except Exception as e:
            print(f"CRAFT detection error: {e}")
            return self._fallback_contour_detect(cv_image)

        rects = []
        if horiz_boxes and len(horiz_boxes) > 0:
            for box in horiz_boxes[0]:
                if len(box) >= 4:
                    x1, x2, y1, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    rects.append([x1, y1, x2, y2])
        if free_boxes and len(free_boxes) > 0:
            for points in free_boxes[0]:
                if len(points) >= 4:
                    xs = [int(p[0]) for p in points if len(p) >= 2]
                    ys = [int(p[1]) for p in points if len(p) >= 2]
                    if xs and ys:
                        rects.append([min(xs), min(ys), max(xs), max(ys)])

        merged = merge_rectangles(rects, threshold=25)
        qrects = []
        for box in merged:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            pad_w, pad_h = int(w * 0.15), int(h * 0.15)
            qrects.append(QRectF(x1 - pad_w, y1 - pad_h, w + 2 * pad_w, h + 2 * pad_h))
        return qrects if qrects else self._fallback_contour_detect(cv_image)

    def _fallback_contour_detect(self, cv_image):
        """Резервный детектор на основе OpenCV контуров белых областей."""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h_img, w_img = cv_image.shape[:2]
        rects = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if 600 < area < (h_img * w_img * 0.4):
                aspect = float(w) / h
                if 0.3 < aspect < 3.0:
                    rects.append(QRectF(x, y, w, h))
        return rects
