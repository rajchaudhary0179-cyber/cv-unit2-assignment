"""
Program 11: Smoothing and Sharpening Comparison
Using the same input image, generates one smoothed output and one
sharpened output for side-by-side comparison.
"""
import cv2
import numpy as np

img = cv2.imread("input.jpg")
if img is None:
    raise FileNotFoundError("input.jpg not found in this folder.")

# Smoothing via Gaussian blur
smoothed = cv2.GaussianBlur(img, (5, 5), sigmaX=0)

# Sharpening via custom kernel
sharpening_kernel = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float32)
sharpened = cv2.filter2D(img, ddepth=-1, kernel=sharpening_kernel)

cv2.imwrite("output_smooth.png", smoothed)
cv2.imwrite("output_sharp.png", sharpened)

print("Saved output_smooth.png (Gaussian blur, 5x5)")
print("Saved output_sharp.png (custom sharpening kernel)")
