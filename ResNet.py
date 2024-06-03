# Import packages and functions
import tensorflow as tf
import pickle
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalMaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from EvaluateModel import evaluate_model
from LoadDataset import load_dataset

# Set seed to have reproducible results
tf.keras.utils.set_random_seed(12345)

# Define paths and classes
data_dir = 'PublicDataset/Train'
classes = ['Naturally-Blurred', 'Undistorted']

# Load the images and image labels into training and validation set
X_train, y_train, X_val, y_val = load_dataset(data_dir, classes)

# Combine the training/validation images and image labels
# Also create batches of size 32
train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)

# Load pre-trained ResNet50 model without the classification layers
base_model = ResNet50(weights='imagenet', include_top=False)

# Add the custom classification layers
x = base_model.output
x = GlobalMaxPooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Create transfer learning model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[tf.keras.metrics.FBetaScore(beta = 0.5, threshold = 0.5)])

# Fit the model
model.fit(train_data, epochs=10, validation_data=val_data)

# Evaluate the model
evaluate_model(model, X_val, y_val)

# Store the model
model_name = 'MaxResNet50.pkl'

with open('Models/' + model_name, 'wb') as file:
    pickle.dump(model, file)