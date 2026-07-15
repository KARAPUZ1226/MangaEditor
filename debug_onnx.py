import onnxruntime
import numpy as np
import cv2
import os

model_path = r"D:\хрень какая-то\MangaEditor\models\custom_detector.onnx"
session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

print("Inputs:")
for inp in session.get_inputs():
    print(inp.name, inp.shape, inp.type)

print("\nOutputs:")
for out in session.get_outputs():
    print(out.name, out.shape, out.type)

# Create a dummy image input of shape (1, 3, 640, 640)
dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: dummy_input})

print("\nOutput[0] shape:", outputs[0].shape)
blk = outputs[0][0]
print("blk shape:", blk.shape)

# Let's print the first 5 elements of each dimension (if shape is 5x8400)
if blk.shape[0] == 5:
    print("Dimensions of blk (5x8400):")
    for i in range(5):
        print(f"Row {i} (first 10 values):", blk[i, :10])
        
# Let's find rows with high confidence in the output
# Wait, let's see which row represents the confidence!
# In YOLOv8, standard output is shape [1, 5, 8400]
# where dim 0 is x_center, dim 1 is y_center, dim 2 is width, dim 3 is height, dim 4 is class_0 score.
# BUT wait! Does YOLOv8 return raw scores or sigmoid-activated scores?
# YOLOv8 ONNX exports sigmoid-activated scores, so they are already between 0.0 and 1.0!
# Let's check if the confidence values are very small or very large.
if blk.shape[0] == 5:
    confidences = blk[4, :]
    print("\nConfidence stats:")
    print("Min:", confidences.min())
    print("Max:", confidences.max())
    print("Mean:", confidences.mean())
    print("Number of confidences > 0.25:", np.sum(confidences > 0.25))
    print("Number of confidences > 0.05:", np.sum(confidences > 0.05))
