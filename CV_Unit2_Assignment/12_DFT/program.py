"""
Program 12: 2D DFT Computation
Reads a grayscale image, computes its 2D DFT using OpenCV, and
shifts the zero-frequency component to the center.
"""
import cv2
import numpy as np

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.jpg not found in this folder.")

# cv2.dft requires a float32 input
img_float32 = np.float32(img)

# Compute the 2D DFT; DFT_COMPLEX_OUTPUT gives a 2-channel (real, imaginary) result
dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)

# Shift the zero-frequency (DC) component from the corners to the center
dft_shifted = np.fft.fftshift(dft)

print("Original image shape:", img.shape)
print("DFT result shape:", dft.shape)
print("Shifted DFT result shape:", dft_shifted.shape)

# Save the shifted DFT data for use by later programs (13, 14, 15)
np.save("dft_shifted.npy", dft_shifted)
print("Saved intermediate DFT data as dft_shifted.npy")

# Also save a basic magnitude visualization as the required output.png
magnitude = cv2.magnitude(dft_shifted[:, :, 0], dft_shifted[:, :, 1])
magnitude_log = 20 * np.log(magnitude + 1)  # +1 avoids log(0)
magnitude_norm = cv2.normalize(magnitude_log, None, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite("output.png", magnitude_norm.astype(np.uint8))
print("Saved DFT magnitude visualization as output.png")
