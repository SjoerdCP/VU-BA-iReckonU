from LoadDataset import load_dataset
from LoadPublicDataset import load_public_dataset
from PreprocessData import preprocess_data
from DefineModel import create_model, fit_model
from EvaluateModel import evaluate_model
from OldCode.Plot import plotHistory

import tensorflow as tf 
import numpy as np

tf.keras.utils.set_random_seed(12345)

blur_folder = './CERTH_ImageBlurDataset/TrainingSet/Naturally-Blurred'
sharp_folder = './CERTH_ImageBlurDataset/TrainingSet/Undistorted'

#X_train, X_val, y_train, y_val = load_dataset('Sharp', 'Blurry')
#X_train, X_val, y_train, y_val = load_public_dataset(sharp_folder, blur_folder)
X_train, X_val, y_train, y_val = load_public_dataset('Sharp', 'Blurry')

X_train = preprocess_data(X_train)
X_val = preprocess_data(X_val)

model = create_model()

epochs = 10

y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)

history = fit_model(model, epochs, X_train, y_train, X_val, y_val)

evaluate_model(model, X_val, y_val)

plotHistory(history)