"""
Program 3: Contrast Stretching
Reads a low-contrast grayscale image, finds its min/max intensity,
then linearly stretches the intensity range to [0, 255].
(This is a manual linear stretch, NOT histogram equalization.)
"""
import cv2
import numpy as np

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.jpg not found in this folder.")

min_val = int(img.min())
max_val = int(img.max())
print("Minimum intensity:", min_val)
print("Maximum intensity:", max_val)

# Linear contrast stretch formula:
# new_pixel = (pixel - min) * (255 / (max - min))
if max_val == min_val:
    stretched = img.copy()
else:
    stretched = (img.astype(np.float32) - min_val) * (255.0 / (max_val - min_val))
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)

print("New minimum intensity:", int(stretched.min()))
print("New maximum intensity:", int(stretched.max()))

cv2.imwrite("output.png", stretched)
print("Saved contrast-stretched result as output.png")
