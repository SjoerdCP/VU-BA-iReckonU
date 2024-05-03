from LoadDataset import load_dataset
from PreprocessData import preprocess_data
from DefineModel import create_model, fit_model
from EvaluateModel import evaluate_model

import tensorflow as tf 

tf.keras.utils.set_random_seed(12345)

X_train, X_val, y_train, y_val = load_dataset('Sharp', 'Blurry')

X_train = preprocess_data(X_train)
X_val = preprocess_data(X_val)

model = create_model()

epochs = 25
history = fit_model(model, epochs, X_train, y_train, X_val, y_val)

evaluate_model(model, X_val, y_val)