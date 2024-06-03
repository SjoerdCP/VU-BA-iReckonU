# Import packages
import cv2
import numpy as np

# Function to preprocess data
def preprocess_data(data):

    # Create array to store images
    images = np.zeros((len(data), 256, 256))

    # Preprocess images
    for i, img in enumerate(data):

        # Convert image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Store image in array
        images[i] = img

    # Normalize pixel intesity values
    images = images / 255.0

    # Return array of images
    return images