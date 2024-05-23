import tensorflow as tf
from LoadPublicDataset import load_public_dataset
from LoadDataset import load_dataset
from EvaluateModel import evaluate_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Set seed
tf.keras.utils.set_random_seed(12345)

# Load pre-trained ResNet50 model without classification layers
base_model = ResNet50(weights='imagenet', include_top=False)

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Create transfer learning model
model = Model(inputs=base_model.input, outputs=predictions)

# Optionally freeze some layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[tf.keras.metrics.FBetaScore(beta = 0.5, threshold = 0.5)])

# Load and preprocess the dataset
blur_folder = './PublicDataset/Train/Naturally-Blurred'
sharp_folder = './PublicDataset/Train/Undistorted'

#X_train, X_val, y_train, y_val = load_public_dataset(sharp_folder, blur_folder)
X_train, X_val, y_train, y_val = load_dataset("./Sharp", "./Blurry")

y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)

# Train the model
train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)

model.fit(train_data, epochs=10, validation_data=val_data)

# # Optionally unfreeze some top layers for fine-tuning
# for layer in model.layers[-50:]:
#     layer.trainable = True

# # Recompile the model after unfreezing layers and adjust the learning rate
# model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=[tf.keras.metrics.FBetaScore(beta = 0.5, threshold = 0.5)])

# # Continue training the model
# model.fit(train_data, epochs=10, validation_data=val_data)

evaluate_model(model, X_val, y_val)