"""
Program 9: Mean vs Gaussian vs Median
Applies Mean, Gaussian, and Median filters to the same noisy input
image and saves each result separately for comparison.
"""
import cv2

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.jpg not found in this folder.")

mean_result = cv2.blur(img, (5, 5))
gaussian_result = cv2.GaussianBlur(img, (5, 5), sigmaX=0)
median_result = cv2.medianBlur(img, 5)

cv2.imwrite("output_mean.png", mean_result)
cv2.imwrite("output_gaussian.png", gaussian_result)
cv2.imwrite("output_median.png", median_result)

print("Saved output_mean.png, output_gaussian.png, output_median.png")
print("Observation: Median filtering removes salt-and-pepper impulse noise")
print("most effectively, while Mean and Gaussian filters blur it instead")
print("of removing it cleanly.")
