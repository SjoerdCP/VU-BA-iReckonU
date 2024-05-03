import cv2
import numpy as np

def preprocess_data(data):
    images = np.zeros((len(data), 256, 256))

    for i, img in enumerate(data):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        images[i] = img

    images.reshape((-1, 256, 256, 1))
    images = images / 255.0

    return images