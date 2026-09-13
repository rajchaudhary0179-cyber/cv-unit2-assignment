"""
Program 13: Magnitude Spectrum
Computes the DFT and centered, log-scaled magnitude spectrum of a
grayscale image, then saves it for visualization.
"""
import cv2
import numpy as np

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.jpg not found in this folder.")

img_float32 = np.float32(img)
dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shifted = np.fft.fftshift(dft)

# Magnitude = sqrt(real^2 + imaginary^2)
magnitude = cv2.magnitude(dft_shifted[:, :, 0], dft_shifted[:, :, 1])

# Log scaling compresses the huge dynamic range of frequency magnitudes
# into a visually interpretable range.
magnitude_log = 20 * np.log(magnitude + 1)  # +1 avoids log(0)

# Normalize to 0-255 for saving as a standard image
magnitude_norm = cv2.normalize(magnitude_log, None, 0, 255, cv2.NORM_MINMAX)
magnitude_uint8 = magnitude_norm.astype(np.uint8)

cv2.imwrite("output.png", magnitude_uint8)
print("Saved centered, log-scaled magnitude spectrum as output.png")
