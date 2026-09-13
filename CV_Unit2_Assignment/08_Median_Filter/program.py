"""
Program 8: Salt-and-Pepper Noise Reduction
Applies median filtering to an image that visibly contains
salt-and-pepper (impulse) noise.
"""
import cv2

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.jpg not found in this folder.")

# Median filtering is very effective against salt-and-pepper noise
# because it replaces each pixel with the median of its neighborhood,
# which discards extreme outlier (0 or 255) impulse values.
KERNEL_SIZE = 5  # must be odd
filtered = cv2.medianBlur(img, KERNEL_SIZE)

cv2.imwrite("output.png", filtered)
print(f"Applied median filter with kernel size {KERNEL_SIZE}.")
print("Saved median-filtered result as output.png")
