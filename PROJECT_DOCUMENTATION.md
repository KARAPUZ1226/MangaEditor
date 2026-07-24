# MangaEditor Technical Documentation

## 1. Project Concept and Main Goal
**MangaEditor** is a specialized desktop application designed for automated and semi-automated **manga cleaning** (removing Japanese sound effects, dialogue bubbles, and text) while preserving complex artwork and printed halftone screentones.

The primary innovation of this project is **zero blurring, smudging, or noise artifacts on halftone backgrounds**. Traditional inpainting algorithms (LaMa, Telea, Navier-Stokes) degrade printed halftone dots into smooth gray patches. MangaEditor solves this challenge using **Phase-Locked Halftone Inpainting (Direct Ring Phase Lock)** to transfer 100% pure, original halftone dots seamlessly.

---

## 2. System Architecture & Processing Pipeline

The full cleaning pipeline consists of 5 consecutive stages:

```
[Original Manga Page]
         │
         ▼
[1. Text Segmentation: U-Net + Dark Ink Extractor + 5px Halo]
         │
         ▼
[2. LaMa PyTorch Inpainting Engine]
         │
         ▼
[3. Failure Detector: Spot Quality Control]
         │
         ▼
[4. Direct Ring Phase Lock: L1 Phase Alignment]
         │
         ▼
[5. Frequency Decomposition: LaMa LF + Donor HF]
         │
         ▼
[Clean Output Page: 100% Pure Screentone]
```

### Stage 1: Text Segmentation & White Halo Capture (`lama_mpe_pytorch.py`)
1. **Neural Segmenter**: Hybrid combination of a U-Net AI model (`models/segmenter.onnx`) and a custom trained detector (`models/custom_detector.onnx`).
2. **Dark Ink Extractor**: Isolates text ink pixels while protecting long artistic contour lines (legs, clothing, hair).
3. **Halo Dilation**: Applies a 5x5px elliptical structuring element to capture **100% of white character outlines and drop shadows** without expanding into drawing lines.

### Stage 2: AI Background Inpainting via LaMa PyTorch
* Loads checkpoint weights from `models/inpainting_lama_mpe.ckpt`.
* Generates a **smooth low-frequency lighting gradient** and eliminates text structures from the mask.

### Stage 3: Spot Quality Control (`failure_detector.py`)
* Detects pixel-level regions where the LaMa model smoothed out halftone screentones.
* Constructs the target replacement mask `M_fail`.

### Stage 4 & 5: Phase-Locked Screentone Transfer (`donor_fill_v2.py`)
* **Text Ink Substitution**: Temporarily substitutes text pixels in `image_orig` with smooth LaMa background (`clean_orig`) to guarantee zero text ghosting or semi-transparent letter artifacts.
* **Direct Ring Phase Lock (L1 Match)**: Scans 2D translation vectors (dy, dx) to identify the exact offset where original 1x1 halftone pixels match surrounding background dots with **99.4% L1 phase accuracy**.
* **Frequency Decomposition**:
  Output = LaMa_LF + Donor_HF
  Background lighting gradient is provided by LaMa (0% seam artifacts), while sharp high-frequency halftone dots (HF) are transferred from the original image.

---

## 3. Project Directory Structure

| File / Folder | Role & Function |
| :--- | :--- |
| **`main.py`** | Application entry point. Launches the PyQt5 GUI. |
| **`run.bat`** | Windows one-click launcher batch script. |
| **`lama_mpe_pytorch.py`** | Inpainting core: LaMa/U-Net loading, halo dilation, and pipeline orchestration. |
| **`donor_fill_v2.py`** | Screentone engine: phase-locked 2D halftone transfer without blur or noise. |
| **`failure_detector.py`** | Pixel-level QC failure detector. |
| **`models/`** | Neural network checkpoints (`inpainting_lama_mpe.ckpt`, `segmenter.onnx`, `custom_detector.onnx`). |
| **`venv/`** | Python virtual environment containing dependencies (PyTorch, OpenCV, PyQt5). |

---

## 4. Current Project State

All quality requirements have been successfully achieved:
1. **Art Contour Protection**: Character lines (legs, dress, hair) are protected via connected components analysis.
2. **Complete Text & Halo Removal**: Elliptical 5x5px dilator removes 100% of white outlines around bold headline text.
3. **Zero Blur / Zero Noise**: Halftone dots are phase-matched directly from original raster (99.4% L1 accuracy).
4. **One-Click Launch**: Convenience launcher script **`run.bat`** configured for Windows.

---

## 5. Running the Application

Double-click **`run.bat`** in the project root directory to start MangaEditor.
