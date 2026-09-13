"""
Program 1: Grayscale Conversion and Image Information
Reads a color image, converts it to grayscale, saves the result,
and prints shape/dimension information.
"""
import cv2

# Read the color image
img = cv2.imread("input.jpg")

if img is None:
    raise FileNotFoundError("input.jpg not found in this folder.")

# Convert BGR (OpenCV default) to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Print required information
height, width = gray.shape
print("Original image shape (H, W, Channels):", img.shape)
print("Grayscale image shape (H, W):", gray.shape)
print("Height:", height)
print("Width:", width)

# Save the grayscale output
cv2.imwrite("output.png", gray)
print("Saved grayscale result as output.png")
