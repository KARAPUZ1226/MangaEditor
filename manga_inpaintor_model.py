import os
import cv2
import numpy as np
import torch

class MangaInpaintorInpainter:
    def __init__(self, model_path="models/manga_inpaintor.jit"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[MangaInpainting] Loading SIGGRAPH 2021 TorchScript checkpoint {model_path} to {self.device}...")
        
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        
        # Load U-Net text segmenter for smart text detection if available
        segmenter_path = os.path.join(os.path.dirname(model_path), "segmenter.onnx")
        if os.path.exists(segmenter_path):
            import onnxruntime as ort
            self.segmenter = ort.InferenceSession(segmenter_path, providers=['CPUExecutionProvider'])
            print(f"[MangaInpainting] Loaded text segmenter from {segmenter_path}")
        else:
            self.segmenter = None

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # BGR (H, W, 3) and mask (H, W) [255 = inpaint]
        height, width = image.shape[:2]
        img_original = np.copy(image)
        gray_orig = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
        
        # 1. Refine mask with U-Net text segmenter & connected components if mask covers broader area
        mask_refined = np.copy(mask)
        y_indices, x_indices = np.where(mask >= 127)
        if len(y_indices) > 0 and self.segmenter is not None:
            try:
                y0_box, y1_box = y_indices.min(), y_indices.max() + 1
                x0_box, x1_box = x_indices.min(), x_indices.max() + 1
                bbox_gray = gray_orig[y0_box:y1_box, x0_box:x1_box]
                bbox_resized = cv2.resize(bbox_gray, (256, 256))
                input_blob = (bbox_resized.astype(np.float32) / 255.0)[None, None, :, :]
                outputs = self.segmenter.run(None, {"input": input_blob})
                probs = 1.0 / (1.0 + np.exp(-outputs[0][0][0]))
                mask_256 = (probs > 0.5).astype(np.uint8) * 255
                bbox_mask = cv2.resize(mask_256, (x1_box - x0_box, y1_box - y0_box), interpolation=cv2.INTER_NEAREST)
                
                unet_mask = np.zeros_like(gray_orig)
                unet_mask[y0_box:y1_box, x0_box:x1_box] = bbox_mask
                
                # Dilate slightly to catch text outlines
                kernel = np.ones((3, 3), np.uint8)
                text_mask_dilated = cv2.dilate(unet_mask, kernel, iterations=4)
                
                # Intersect with user selection
                mask_refined[text_mask_dilated == 0] = 0
            except Exception as e:
                print(f"[MangaInpainting] Segmenter refinement skipped: {e}")

        # Ensure height and width are divisible by 8 using reflection padding
        pad_size = 8
        pad_h = (pad_size - (height % pad_size)) % pad_size
        pad_w = (pad_size - (width % pad_size)) % pad_size
        
        if pad_h > 0 or pad_w > 0:
            gray_padded = cv2.copyMakeBorder(gray_orig, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            mask_padded = cv2.copyMakeBorder(mask_refined, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        else:
            gray_padded = gray_orig
            mask_padded = mask_refined
            
        ph, pw = gray_padded.shape[:2]

        # 2. Extract structural line art (Canny edges inverted)
        edges = cv2.Canny(gray_padded, 50, 150)
        line_art = 255 - edges
        
        # 3. Prepare Tensors
        img_t = torch.from_numpy((gray_padded.astype(np.float32) / 127.5) - 1.0).unsqueeze(0).unsqueeze(0).to(self.device)
        lines_t = torch.from_numpy((line_art.astype(np.float32) / 127.5) - 1.0).unsqueeze(0).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy((mask_padded.astype(np.float32) / 255.0)).unsqueeze(0).unsqueeze(0).to(self.device)
        mask_t[mask_t < 0.5] = 0
        mask_t[mask_t >= 0.5] = 1

        noise_t = torch.randn(1, 1, ph, pw).to(self.device)
        ones_t = torch.ones(1, 1, ph, pw).to(self.device)

        # 4. Neural Inpainting
        with torch.no_grad():
            out_t = self.model(img_t, lines_t, mask_t, noise_t, ones_t)
            out_gray = ((out_t.squeeze(0).squeeze(0).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

        # Unpad
        if pad_h > 0 or pad_w > 0:
            out_gray = out_gray[0:height, 0:width]

        # Convert to BGR
        out_bgr = cv2.cvtColor(out_gray, cv2.COLOR_GRAY2BGR)

        # Seamless blend with feathered mask
        mask_float = (mask_refined.astype(np.float32) / 255.0)[:, :, None]
        feathered = cv2.GaussianBlur(mask_float, (7, 7), 0)
        if len(feathered.shape) == 2:
            feathered = feathered[:, :, None]

        final_ans = out_bgr.astype(np.float32) * feathered + img_original.astype(np.float32) * (1.0 - feathered)
        return np.clip(final_ans, 0, 255).astype(np.uint8)
