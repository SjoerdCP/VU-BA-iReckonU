# Import packages and functions
import cv2
import numpy as np
from sklearn.metrics import fbeta_score
from scipy.optimize import dual_annealing
from PreprocessData import preprocess_data

# Function to get the laplacian variance of an image
def laplacian_var(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

# The laplacian model class
class LaplaceModel:

    # Initialize untrained model
    def __init__(self):
        self.threshold = -1

    # Train model
    def fit(self, X, y):

        # Function to minimize
        def minimize(threshold, vars, y):
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
    
    # Function to predict image labels
    def predict(self, X):
        # Preprocess data
        X = preprocess_data(X)
        
        # Predict image labels
        y_pred = np.array([1 if laplacian_var(img) < self.threshold else 0 for img in X])
        
        # Return predicted image labels
        return y_pred