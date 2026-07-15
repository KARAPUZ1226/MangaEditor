import cv2
import os
import sys
import numpy as np

# Add current path
sys.path.insert(0, os.path.abspath("."))
from translator import ComicTextDetector, EasyOCRManager

detector = ComicTextDetector()
detector.load()
manager = EasyOCRManager()

img_path = r"D:\Загрузки\Новая папка\dungeon-camper-no-ore-gal-haishinsha-wo-tasuketara-bazutta-ue-ni-mainichi-gal-ga-meshi-wo-kui-ni-kuru-chapter-7\001.jpg"
img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
h_orig, w_orig = img.shape[:2]

raw_results = detector.detect(img)
print(f"Total raw boxes found: {len(raw_results)}")

for i, qr in enumerate(raw_results):
    x1, y1 = int(qr.x()), int(qr.y())
    w, h = int(qr.width()), int(qr.height())
    
    x1 = max(0, min(x1, w_orig - 1))
    y1 = max(0, min(y1, h_orig - 1))
    w = min(w, w_orig - x1)
    h = min(h, h_orig - y1)
    
    crop = img[y1:y1+h, x1:x1+w]
    
    from PIL import Image
    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    mocr = manager.get_manga_ocr()
    text = mocr(pil_img) if mocr else ""
    
    is_unwanted = manager._is_unwanted_text(text, 'ja')
    
    print(f"Box {i} (box={[x1,y1,x1+w,y1+h]}):")
    print(f"  OCR Text: '{text}'")
    print(f"  Is Unwanted: {is_unwanted}")
