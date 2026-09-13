"""
Program 2: Controlled Brightness Enhancement
Reads a dark image and increases brightness by a constant value,
clipping pixel values so they stay within the valid 0-255 range.
"""
import cv2
import numpy as np

img = cv2.imread("input.jpg")
if img is None:
    raise FileNotFoundError("input.jpg not found in this folder.")

BRIGHTNESS_INCREASE = 80  # constant added to every pixel

# Pick a sample pixel to compare before/after
sample_y, sample_x = img.shape[0] // 2, img.shape[1] // 2
before_pixel = img[sample_y, sample_x].copy()

# Use int16 during addition to avoid uint8 overflow/wraparound,
# then clip back into the valid [0, 255] range before converting back.
brightened = img.astype(np.int16) + BRIGHTNESS_INCREASE
brightened = np.clip(brightened, 0, 255).astype(np.uint8)

after_pixel = brightened[sample_y, sample_x]

print(f"Pixel at ({sample_y},{sample_x}) before enhancement: {before_pixel}")
print(f"Pixel at ({sample_y},{sample_x}) after  enhancement: {after_pixel}")

cv2.imwrite("output.png", brightened)
print("Saved brightened result as output.png")
