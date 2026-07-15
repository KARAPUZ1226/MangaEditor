import os
import cv2
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath("."))
from translator import EasyOCRManager

manager = EasyOCRManager()
mocr = manager.get_manga_ocr()
if not mocr:
    print("MangaOCR not available.")
    sys.exit(1)

ch7_dir = r"D:\Загрузки\Новая папка\dungeon-camper-no-ore-gal-haishinsha-wo-tasuketara-bazutta-ue-ni-mainichi-gal-ga-meshi-wo-kui-ni-kuru-chapter-7"
for f in sorted(os.listdir(ch7_dir)):
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        fp = os.path.join(ch7_dir, f)
        img = cv2.imdecode(np.fromfile(fp, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        # Check OCR of the whole image or sub-regions
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        text = mocr(pil_img)
        if "配信者" in text or "こんにちは" in text:
            print(f"Found in {f}: {text}")
            break
