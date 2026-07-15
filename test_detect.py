import cv2
import os
import sys
import numpy as np

# Add current path
sys.path.insert(0, os.path.abspath("."))
from translator import ComicTextDetector

detector = ComicTextDetector()
if not detector.load():
    print("Could not load detector.")
    sys.exit(1)

# Find any image in the chapter directory
img_path = r"D:\хрень какая-то\Рабыня\jibun-wo-oshiuri-shite-kita-dorei-chan-ga-dragon-wo-one-punch-shiteta-chapter-8\001.jpg"

if not os.path.exists(img_path):
    print(f"File not found: {img_path}")
    sys.exit(1)

# Read using imdecode to support cyrillic paths
img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
print("Loaded image shape:", img.shape)

results = detector.detect(img)
print("Number of detected bubbles:", len(results))
for i, r in enumerate(results[:10]):
    # Note: ComicTextDetector.detect returns a list of dictionaries or objects?
    # Actually, in translator.py, it returns a list of dicts: [{'box': [x1,y1,x2,y2], 'class': ...}] or similar
    print(f"Bubble {i}: {r}")
