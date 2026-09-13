"""
Program 7: Gaussian Smoothing
Applies Gaussian smoothing to a noisy image using an odd kernel size.
"""
import cv2

img = cv2.imread("input.jpg")
if img is None:
    raise FileNotFoundError("input.jpg not found in this folder.")

# A 5x5 kernel is chosen because it gives noticeable noise reduction
# while still preserving most edge detail. Kernel sizes must be odd
# so that there is a well-defined center pixel; sigma=0 lets OpenCV
# auto-compute the standard deviation from the kernel size.
KERNEL_SIZE = (5, 5)
smoothed = cv2.GaussianBlur(img, KERNEL_SIZE, sigmaX=0)

cv2.imwrite("output.png", smoothed)
print(f"Applied Gaussian blur with kernel {KERNEL_SIZE}.")
print("Saved Gaussian-smoothed result as output.png")
