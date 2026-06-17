import cv2

img = cv2.imread("images/sample.jpg")

resized = cv2.resize(img, (224, 224))

print("Original Shape:", img.shape)
print("Resized Shape:", resized.shape)
