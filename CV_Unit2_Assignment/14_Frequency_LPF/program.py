"""
Program 14: Frequency-Domain Low-Pass Filtering
Creates a circular low-pass mask that keeps only the central
low-frequency region, applies it in the frequency domain, and
reconstructs the filtered spatial-domain image.
"""
import cv2
import numpy as np

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.jpg not found in this folder.")

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

# Forward DFT and shift so low frequencies are centered
img_float32 = np.float32(img)
dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shifted = np.fft.fftshift(dft)

# Circular low-pass mask: 1 inside the radius (keep low frequencies),
# 0 outside (suppress high frequencies).
RADIUS = 40
mask = np.zeros((rows, cols, 2), np.float32)
y, x = np.ogrid[:rows, :cols]
dist_from_center = (x - ccol) ** 2 + (y - crow) ** 2
mask_area = dist_from_center <= RADIUS ** 2
mask[mask_area] = 1

# Apply mask to the shifted spectrum
filtered_dft = dft_shifted * mask

# Inverse shift, then inverse DFT to go back to the spatial domain
filtered_dft_unshifted = np.fft.ifftshift(filtered_dft)
img_reconstructed = cv2.idft(filtered_dft_unshifted)
img_reconstructed = cv2.magnitude(img_reconstructed[:, :, 0], img_reconstructed[:, :, 1])

# Normalize back to a displayable 0-255 range
img_reconstructed = cv2.normalize(img_reconstructed, None, 0, 255, cv2.NORM_MINMAX)
img_reconstructed = img_reconstructed.astype(np.uint8)

cv2.imwrite("output.png", img_reconstructed)
print(f"Applied circular low-pass mask (radius={RADIUS}).")
print("Saved low-pass filtered (blurred) reconstruction as output.png")
