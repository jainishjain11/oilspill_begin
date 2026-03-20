import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import cv2

# Paths
BASE_DIR = r"D:\oilspill_project\data\data"
OIL_DIR = os.path.join(BASE_DIR, "S1SAR_UnBalanced_400by400_Class_1", "1")
NO_OIL_DIR = os.path.join(BASE_DIR, "S1SAR_UnBalanced_400by400_Class_0", "0")

print("🔄 Loading dataset...")

# Load images (balanced subset for speed)
oil_files = os.listdir(OIL_DIR)[:500]  # First 500
no_oil_files = os.listdir(NO_OIL_DIR)[:500]

def load_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img = np.stack([img]*3, axis=-1)  # Grayscale → RGB for MobileNet
    img = img.astype(np.float32) / 255.0
    return img

# Load data
X_oil = np.array([load_image(os.path.join(OIL_DIR, f)) for f in oil_files])
X_nooil = np.array([load_image(os.path.join(NO_OIL_DIR, f)) for f in no_oil_files])

X = np.concatenate([X_oil, X_nooil])
y = np.concatenate([np.ones(500), np.zeros(500)])  # 1=oil, 0=no-oil

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✅ Dataset: {X_train.shape[0]} train, {X_test.shape[0]} test")

# MobileNetV2 Transfer Learning
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

print("🚀 Training MobileNetV2...")
history = model.fit(X_train, y_train, epochs=5, batch_size=32, 
                   validation_data=(X_test, y_test), verbose=1)

# Test on sample images
sample_oil = X_oil[0:1]
sample_nooil = X_nooil[0:1]

oil_pred = model.predict(sample_oil)[0][0]
nooil_pred = model.predict(sample_nooil)[0][0]

print(f"📊 Oil prediction: {oil_pred:.3f} (should be >0.5)")
print(f"📊 No-oil prediction: {nooil_pred:.3f} (should be <0.5)")

# Plot results
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.title(f"Oil SAR\nPred: {oil_pred:.3f}")
plt.imshow(sample_oil[0][:,:,0], cmap="gray")

plt.subplot(1, 4, 2)
plt.title(f"No-oil SAR\nPred: {nooil_pred:.3f}")
plt.imshow(sample_nooil[0][:,:,0], cmap="gray")

plt.subplot(1, 4, 3)
plt.title("Training Accuracy")
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.legend()

plt.subplot(1, 4, 4)
plt.title("Training Loss")
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.legend()

plt.tight_layout()
plt.savefig("mobilenet_results.png")
plt.show()

print("🎉 TRANSFER LEARNING COMPLETE!")
print("📊 Results saved as 'mobilenet_results.png'")
model.save("mobilenetv2_oil_model.h5")
print("💾 Model saved as 'mobilenetv2_oil_model.h5'")

#No fuzzy needed in CNN (MobileNetV2 handles noise internally)