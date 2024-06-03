# Import packages and functions
from LoadDataset import load_dataset
from PreprocessData import preprocess_data
from EvaluateModel import evaluate_model
from LaplaceModel import LaplaceModel
import tensorflow as tf 
import pickle

# Set seed to have reproducible results
tf.keras.utils.set_random_seed(12345)

# Set directory and classes
data_dir = 'PublicDataset/Train'
classes = ['Naturally-Blurred', 'Undistorted']

# Load data
X_train, y_train, X_val, y_val = load_dataset(data_dir, classes)
print('Loaded Dataset')

# Create and fit laplacian model
model = LaplaceModel()
model.fit(X_train, y_train)

# Evaluate model on validation set
evaluate_model(model, X_val, y_val)

# Store model using pickle
model_name = 'public_laplacian.pkl'

with open('Models/' + model_name, 'wb') as file:
    pickle.dump(model, file)