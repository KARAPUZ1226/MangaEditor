import os
import cv2
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath("."))
from translator import EasyOCRManager

manager = EasyOCRManager()
mocr = manager.get_manga_ocr()

base_dir = r"D:\Загрузки\Новая папка"
found = False

for root, dirs, files in os.walk(base_dir):
    for f in sorted(files):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            fp = os.path.join(root, f)
            img = cv2.imdecode(np.fromfile(fp, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            text = mocr(pil_img)
            if "配信者" in text or "岡本" in text or "えっと" in text:
                print(f"Found match: path={fp}, OCR={text}")
                found = True
                break
    if found:
        break
