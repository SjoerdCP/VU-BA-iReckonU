# Import packages and functions
import cv2
import numpy as np
from sklearn.metrics import fbeta_score
from scipy.optimize import dual_annealing

def preprocess_data(data):
    """
    Preprocess the input data by converting images to grayscale, resizing them, 
    and normalizing pixel intensity values.

    Parameters:
    data (array-like): List or array of images.

    Returns:
    numpy.ndarray: Array of preprocessed images.
    """

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

def laplacian_var(img):
    """
    Calculate the Laplacian variance of an image. This function calculates the variance 
    of the Laplacian of the image, which is a measure of the image's edge strength.

    Parameters:
    image (numpy.ndarray): Grayscale image.

    Returns:
    float: Laplacian variance of the image.
    """

    # Calculate and return the Laplacian variance of an image
    return cv2.Laplacian(img, cv2.CV_64F).var()

class LaplaceModel:
    """
    LaplaceModel is a class that implements a binary classification model based on 
    the Laplacian variance of images. The model is trained by optimizing a threshold 
    that minimizes the F-beta score.
    """

    def __init__(self):
        """
        Initialize the untrained model with a default threshold value.
        """

        self.threshold = -1

    def fit(self, X, y):
        """
        Train the model by finding the optimal threshold that minimizes the F-beta score.

        Parameters:
        X (array-like): Training data, a list or array of images.
        y (array-like): Target labels, a list or array of binary labels.
        """
        
        def minimize(threshold, vars, y):
            """
            Objective function to minimize. It calculates the negative F-beta score 
            given a threshold.
            """

            y_pred = [1 if var < threshold else 0 for var in vars]
            return -fbeta_score(y, y_pred, beta=0.5)
        
        # Preprocess data
        X = preprocess_data(X)

        # Calculate laplacian variance for every image
        vars = [laplacian_var(x) for x in X]

        # Set starting value for the threshold
        threshold = [0]

        # Optimize the threshold to minimize the minimize function using dual_annealing
        res = dual_annealing(minimize, bounds = list(zip([0], [1])), x0= threshold, args = (vars, y))

        # Set the threshold of the model to the optimized value
        self.threshold = res.x[0]
    
    def predict(self, X):
        """
        Predict labels for the input data using the trained model.

        Parameters:
        X (array-like): Input data, a list or array of images.

        Returns:
        array: Predicted binary labels for the input data.
        """

        # Preprocess data
        X = preprocess_data(X)
        
        # Predict image labels
        y_pred = np.array([1 if laplacian_var(img) < self.threshold else 0 for img in X])
        
        # Return predicted image labels
        return y_pred