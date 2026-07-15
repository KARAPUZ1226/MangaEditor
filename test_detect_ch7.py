import cv2
import os
import sys
import numpy as np

# Add current path
sys.path.insert(0, os.path.abspath("."))
from translator import ComicTextDetector, EasyOCRManager

detector = ComicTextDetector()
if not detector.load():
    print("Could not load detector.")
    sys.exit(1)

manager = EasyOCRManager()

# Target image is Chapter 7 Page 1
img_path = r"D:\Загрузки\Новая папка\dungeon-camper-no-ore-gal-haishinsha-wo-tasuketara-bazutta-ue-ni-mainichi-gal-ga-meshi-wo-kui-ni-kuru-chapter-7\001.jpg"

if not os.path.exists(img_path):
    # Try webp or other files in ch7
    ch7_dir = r"D:\Загрузки\Новая папка\dungeon-camper-no-ore-gal-haishinsha-wo-tasuketara-bazutta-ue-ni-mainichi-gal-ga-meshi-wo-kui-ni-kuru-chapter-7"
    if os.path.exists(ch7_dir):
        files = os.listdir(ch7_dir)
        print("Files in ch7 folder:", files[:5])
        img_path = os.path.join(ch7_dir, files[0])
    else:
        print(f"Ch7 folder not found: {ch7_dir}")
        sys.exit(1)

img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
print("Loaded image shape:", img.shape)

# Run raw detector
raw_results = detector.detect(img)
print(f"Raw detector found {len(raw_results)} boxes:")
for i, r in enumerate(raw_results):
    print(f"Raw {i}: box={r}")

# Run filtered detector_ai
filtered = manager.detect_bubbles_ai(img, 'ja')
print(f"Filtered detector_ai returned {len(filtered)} boxes:")
for i, r in enumerate(filtered):
    print(f"Filtered {i}: box={r}")
