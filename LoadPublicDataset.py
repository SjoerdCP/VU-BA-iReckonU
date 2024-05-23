import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder, label):
    images = []
    labels = []
        
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            img = cv2.resize(img, (256, 256))  # Resize to 256x256
            images.append(img)
            labels.append(label)
    return images, labels

def load_public_dataset(folder_sharp, folder_blurry):
    sharp_images, sharp_labels = load_images_from_folder(folder_sharp, 0)  # Label 1 for sharp
    blurry_images, blurry_labels = load_images_from_folder(folder_blurry, 1)  # Label 0 for blurry

    # More sharp than blurred
    sharp_indices = np.random.choice(len(sharp_images), size=len(blurry_images), replace=False)

    sharp_images = np.array(sharp_images)
    sharp_labels = np.array(sharp_labels)

    sharp_imgs = list(sharp_images[sharp_indices])
    sharp_labs = list(sharp_labels[sharp_indices])

    X = sharp_imgs + blurry_images
    y = sharp_labs + blurry_labels

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Split data into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, shuffle = True, random_state=42)

    return (X_train, X_val, y_train, y_val)
    # Now X_train, X_val, y_train, and y_val can be used for model training and validation