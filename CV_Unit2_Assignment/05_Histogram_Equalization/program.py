"""
Program 5: Histogram Equalization with Before/After Comparison
Performs histogram equalization on a grayscale image, saves the
equalized image, and saves a comparison plot of both histograms.
"""
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.jpg not found in this folder.")

equalized = cv2.equalizeHist(img)

# Save the equalized image
cv2.imwrite("output.png", equalized)
print("Saved equalized image as output.png")

# Compute histograms for before/after
hist_before = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
hist_after = cv2.calcHist([equalized], [0], None, [256], [0, 256]).flatten()

# Save a single comparison plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(hist_before, color="red")
axes[0].set_title("Histogram Before Equalization")
axes[0].set_xlabel("Pixel Intensity")
axes[0].set_ylabel("Frequency")
axes[0].set_xlim([0, 256])
axes[0].grid(alpha=0.3)

axes[1].plot(hist_after, color="blue")
axes[1].set_title("Histogram After Equalization")
axes[1].set_xlabel("Pixel Intensity")
axes[1].set_ylabel("Frequency")
axes[1].set_xlim([0, 256])
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("histogram_comparison.png", bbox_inches="tight")
plt.close()
print("Saved comparison plot as histogram_comparison.png")
