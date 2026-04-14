import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("cancer_model.h5")

IMG_SIZE = 224

def predict_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        return "Error: Image not found or path incorrect"

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    prediction = model.predict(img)

    if prediction > 0.5:
        return "Tumor Detected"
    else:
        return "No Tumor"

# Test
print(predict_image("test.jpg"))