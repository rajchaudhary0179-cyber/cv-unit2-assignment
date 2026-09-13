"""
Program 15: Frequency-Domain High-Pass Filtering
Creates a circular high-pass mask that suppresses the central
low-frequency region and preserves higher frequency content
(edges/fine detail), then reconstructs the spatial-domain image.
"""
import cv2
import numpy as np

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.jpg not found in this folder.")

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

img_float32 = np.float32(img)
dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shifted = np.fft.fftshift(dft)

# Circular high-pass mask: 0 inside the radius (suppress low frequencies),
# 1 outside (keep high frequencies / edges).
RADIUS = 40
mask = np.ones((rows, cols, 2), np.float32)
y, x = np.ogrid[:rows, :cols]
dist_from_center = (x - ccol) ** 2 + (y - crow) ** 2
mask_area = dist_from_center <= RADIUS ** 2
mask[mask_area] = 0

filtered_dft = dft_shifted * mask

filtered_dft_unshifted = np.fft.ifftshift(filtered_dft)
img_reconstructed = cv2.idft(filtered_dft_unshifted)
img_reconstructed = cv2.magnitude(img_reconstructed[:, :, 0], img_reconstructed[:, :, 1])

img_reconstructed = cv2.normalize(img_reconstructed, None, 0, 255, cv2.NORM_MINMAX)
img_reconstructed = img_reconstructed.astype(np.uint8)

cv2.imwrite("output.png", img_reconstructed)
print(f"Applied circular high-pass mask (radius={RADIUS}).")
print("Saved high-pass filtered (edge-emphasized) reconstruction as output.png")
