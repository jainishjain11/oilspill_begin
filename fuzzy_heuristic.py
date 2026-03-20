import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Your paths
BASE_DIR = r"D:\oilspill_project\data\data"
OIL_DIR = os.path.join(BASE_DIR, "S1SAR_UnBalanced_400by400_Class_1", "1")
NO_OIL_DIR = os.path.join(BASE_DIR, "S1SAR_UnBalanced_400by400_Class_0", "0")

def fuzzy_denoise_numpy(img):
    """
    Pure NumPy fuzzy denoising (no external libs)
    """
    img_float = img.astype(np.float32) / 255.0
    
    # Gaussian membership: low intensity = 1 (oil), high = 0 (noise)
    sigma, mean = 0.15, 0.15
    low_mf = np.exp(-((img_float - mean)**2) / (2 * sigma**2))
    
    # Denoised = membership * intensity (keep oil, suppress noise)
    denoised = (low_mf * img_float * 255).astype(np.uint8)
    
    return denoised

def detect_oil_pipeline(img):
    """
    Complete pipeline: Fuzzy → Threshold → Morphology
    """
    # 1. Fuzzy denoising
    img_denoised = fuzzy_denoise_numpy(img)
    
    # 2. Threshold dark regions
    _, thresh = cv2.threshold(img_denoised, 70, 255, cv2.THRESH_BINARY_INV)
    
    # 3. Morphology cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    
    # 4. Contour filtering
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:  # Minimum oil spill size
            cv2.fillPoly(mask, [cnt], 255)
    
    return img_denoised, mask

# Test pipeline
print("🔍 Testing approximate learning pipeline...")

oil_files = os.listdir(OIL_DIR)
no_oil_files = os.listdir(NO_OIL_DIR)

# First images
oil_path = os.path.join(OIL_DIR, oil_files[0])
no_oil_path = os.path.join(NO_OIL_DIR, no_oil_files[0])

oil_img = cv2.imread(oil_path, cv2.IMREAD_GRAYSCALE)
no_oil_img = cv2.imread(no_oil_path, cv2.IMREAD_GRAYSCALE)

print(f"Testing: {oil_files[0]} (oil) vs {no_oil_files[0]} (no-oil)")

# Run detection
oil_denoised, oil_mask = detect_oil_pipeline(oil_img)
no_oil_denoised, no_oil_mask = detect_oil_pipeline(no_oil_img)

# Results
print(f"✅ Oil detection area: {np.sum(oil_mask > 0)} pixels")
print(f"✅ No-oil false positives: {np.sum(no_oil_mask > 0)} pixels")

# Plot
plt.figure(figsize=(16, 10))
titles = ["Input", "Fuzzy Denoised", "Oil Spill Detection"]

plt.subplot(2, 3, 1); plt.title("Oil " + titles[0]); plt.imshow(oil_img, cmap="gray")
plt.subplot(2, 3, 2); plt.title("Oil " + titles[1]); plt.imshow(oil_denoised, cmap="gray")
plt.subplot(2, 3, 3); plt.title("Oil " + titles[2]); plt.imshow(oil_mask, cmap="gray")

plt.subplot(2, 3, 4); plt.title("No-oil " + titles[0]); plt.imshow(no_oil_img, cmap="gray")
plt.subplot(2, 3, 5); plt.title("No-oil " + titles[1]); plt.imshow(no_oil_denoised, cmap="gray")
plt.subplot(2, 3, 6); plt.title("No-oil " + titles[2]); plt.imshow(no_oil_mask, cmap="gray")

plt.tight_layout()
plt.savefig("fuzzyimage.png")  # Save for your report!
plt.show()

print("🎉Approximate learning pipeline working!")
print("📊 Screenshot saved as 'fuzzyimage.png'")
