# Import packages and functions
from LoadDataset import load_train_val_dataset
from LaplaceModel import LaplaceModel
import tensorflow as tf 
import pickle
import numpy as np
import os

"""
TrainLaplacian.py is the script that is used to create and train a Laplacian Variance based model to classify blurred (1) and sharp (0) images
"""

# Set directory and classes ([blurred class, sharp class])
data_dir = 'PublicDataset/Train'
classes = ['Naturally-Blurred', 'Undistorted']

# Set the model directory and name
model_dir = 'Models'
model_name = 'Laplacian.pkl'

# Set seed to have reproducible results
tf.keras.utils.set_random_seed(12345)

# Load data
X_train, y_train, X_val, y_val = load_train_val_dataset(data_dir, classes)

# Create and fit laplacian model
model = LaplaceModel()
model.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))

# Get the path where to store the model
model_path = os.path.join(model_dir, model_name)

# Store the model
with open(model_path, 'wb') as file:
    pickle.dump(model, file)

# Print the threshold
print(f'The threshold of the model is: {model.threshold}')