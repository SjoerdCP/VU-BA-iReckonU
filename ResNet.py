import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.utils import resample

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from EvaluateModel import evaluate_model

# Set seed
tf.keras.utils.set_random_seed(12345)

# Define paths
data_dir = 'PublicDataset/Train'
categories = ['Naturally-Blurred', 'Undistorted']

# Create a DataFrame with file paths and labels
file_paths = []
labels = []

for category in categories:
    category_path = os.path.join(data_dir, category)
    for file_name in os.listdir(category_path):
        file_paths.append(os.path.join(category_path, file_name))
        labels.append(category)

df = pd.DataFrame({'file_path': file_paths, 'label': labels})

# Undersample the majority class in the training data
# Separate majority and minority classes
df_majority = df[df['label'] == 'Undistorted']
df_minority = df[df['label'] == 'Naturally-Blurred']

# Undersample majority class
df_majority_undersampled = resample(df_majority,
                                    replace=False,     # Sample without replacement
                                    n_samples=len(df_minority))    # Match number of samples in minority class

# Combine minority class with undersampled majority class
df_undersampled = pd.concat([df_minority, df_majority_undersampled])

# Perform stratified splitting
train_df, val_df = train_test_split(df_undersampled	, test_size=0.2, stratify=df_undersampled['label'])

# Parameters
img_height = 256
img_width = 256

def load_images(df):
    images = []
    labels = []
    for index, row in df.iterrows():
        img = load_img(row['file_path'], target_size=(img_height, img_width))
        img_array = img_to_array(img)
        images.append(img_array)
        labels.append(row['label'])
    return np.array(images), np.array(labels)

# Load training images and labels
X_train, y_train = load_images(train_df)

# # Load validation images and labels
X_val, y_val = load_images(val_df)

# # Convert labels to binary (0 and 1)
label_to_index = {'Naturally-Blurred': 1, 'Undistorted': 0}
y_train = np.array([label_to_index[label] for label in y_train])
y_val = np.array([label_to_index[label] for label in y_val])

y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)

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