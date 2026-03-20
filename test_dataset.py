import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Your exact paths
BASE_DIR = r"D:\oilspill_project\data\data"
OIL_DIR = os.path.join(BASE_DIR, "S1SAR_UnBalanced_400by400_Class_1", "1")
NO_OIL_DIR = os.path.join(BASE_DIR, "S1SAR_UnBalanced_400by400_Class_0", "0")

# Get first image from each
oil_files = os.listdir(OIL_DIR)
no_oil_files = os.listdir(NO_OIL_DIR)

oil_path = os.path.join(OIL_DIR, oil_files[0])
no_oil_path = os.path.join(NO_OIL_DIR, no_oil_files[0])

print("Testing:", oil_files[0], "and", no_oil_files[0])

# Load images
oil_img = cv2.imread(oil_path, cv2.IMREAD_GRAYSCALE)
no_oil_img = cv2.imread(no_oil_path, cv2.IMREAD_GRAYSCALE)

print("Oil shape:", oil_img.shape, "Mean intensity:", np.mean(oil_img))
print("No-oil shape:", no_oil_img.shape, "Mean intensity:", np.mean(no_oil_img))

# Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.title("Oil spill SAR")
plt.imshow(oil_img, cmap="gray")

plt.subplot(1, 3, 2)
plt.title("No-oil SAR")
plt.imshow(no_oil_img, cmap="gray")

plt.subplot(1, 3, 3)
plt.title("Intensity histogram")
plt.hist(oil_img.flatten(), bins=50, alpha=0.7, label="Oil", color="red")
plt.hist(no_oil_img.flatten(), bins=50, alpha=0.7, label="No-oil", color="blue")
plt.legend()
plt.tight_layout()
plt.show()
