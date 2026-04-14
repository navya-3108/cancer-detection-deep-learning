import cv2
import numpy as np
import os

IMG_SIZE = 224

def load_data(dataset_path):
    data = []
    labels = []

    for category in ["yes", "no"]:
        path = os.path.join(dataset_path, category)
        label = 1 if category == "yes" else 0

        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                image = cv2.imread(img_path)
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                image = image / 255.0

                data.append(image)
                labels.append(label)
            except:
                pass

    return np.array(data), np.array(labels)