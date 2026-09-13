"""
Program 4: Histogram Analysis
Reads a grayscale image, computes and plots its intensity histogram,
saves the plot, and prints the most frequent intensity value.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.jpg not found in this folder.")

# Compute histogram: 256 bins for intensities 0-255
hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()

most_frequent_value = int(np.argmax(hist))
print("Intensity value with highest frequency:", most_frequent_value)
print("Frequency (pixel count) at that value:", int(hist[most_frequent_value]))

plt.figure(figsize=(8, 5))
plt.plot(hist, color="black")
plt.title("Grayscale Intensity Histogram")
plt.xlabel("Pixel Intensity (0-255)")
plt.ylabel("Frequency")
plt.xlim([0, 256])
plt.grid(alpha=0.3)

plt.savefig("output.png", bbox_inches="tight")
plt.close()
print("Saved histogram as output.png")
