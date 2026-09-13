"""
Program 10: Image Sharpening Using a Custom Kernel
Creates a sharpening kernel and applies it using cv2.filter2D().
"""
import cv2
import numpy as np

img = cv2.imread("input.jpg")
if img is None:
    raise FileNotFoundError("input.jpg not found in this folder.")

# Classic sharpening kernel: boosts the center pixel while subtracting
# the four direct neighbors, which emphasizes edges/high-frequency detail.
sharpening_kernel = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float32)

sharpened = cv2.filter2D(img, ddepth=-1, kernel=sharpening_kernel)

cv2.imwrite("output.png", sharpened)
print("Applied custom sharpening kernel:\n", sharpening_kernel)
print("Saved sharpened result as output.png")
