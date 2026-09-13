"""
Program 6: Mean Filtering
Applies a mean/average filter to a noisy image using two different
kernel sizes, and saves the result produced by the larger kernel.
"""
import cv2

img = cv2.imread("input.jpg")
if img is None:
    raise FileNotFoundError("input.jpg not found in this folder.")

# Try a smaller kernel first (less smoothing, testing only)
small_kernel = (3, 3)
smoothed_small = cv2.blur(img, small_kernel)
print(f"Applied mean filter with kernel {small_kernel} (test run).")

# Larger kernel: gives stronger smoothing, this is our required final output
large_kernel = (7, 7)
smoothed_large = cv2.blur(img, large_kernel)
print(f"Applied mean filter with kernel {large_kernel} (final output).")

# Save only the result using the larger kernel, as required
cv2.imwrite("output.png", smoothed_large)
print("Saved mean-filtered result (7x7 kernel) as output.png")
