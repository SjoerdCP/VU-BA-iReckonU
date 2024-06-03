import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from EvaluateModel import evaluate_model
from LoadDataset import load_dataset_new
# Set seed
tf.keras.utils.set_random_seed(12345)

# Define paths
data_dir = 'PublicDataset/Train'
categories = ['Naturally-Blurred', 'Undistorted']

# data_dir = ''
# categories = ['Blurry', 'Sharp']


X_train, y_train, X_val, y_val = load_dataset_new(data_dir, categories)

train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)
# Load pre-trained ResNet50 model without classification layers
base_model = ResNet50(weights='imagenet', include_top=False)

# Add custom classification layers
x = base_model.output
x = GlobalMaxPooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Create transfer learning model
model = Model(inputs=base_model.input, outputs=predictions)

# Optionally freeze some layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[tf.keras.metrics.FBetaScore(beta = 0.5, threshold = 0.5)])

model.fit(train_data, epochs=10, validation_data=val_data)

evaluate_model(model, X_val, y_val)

model.save('TestResNet.keras')