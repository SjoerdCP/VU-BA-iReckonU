# Import packages and functions
import tensorflow as tf
import pickle
import os
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152
from tensorflow.keras.layers import Dense, GlobalMaxPooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from LoadDataset import load_train_val_dataset
import matplotlib.pyplot as plt

# Set seed to have reproducible results
tf.keras.utils.set_random_seed(12345)

# Define paths and classes
dir = 'AugmentedDataset'
classes = ['Blurry', 'Sharp']

# Store the model
model_dir = 'Models'
model_name = 'ResNet50.pkl'

# Model type: one of 'ResNet50', 'ResNet101', 'ResNet152'
model_type = 'ResNet50'

try:
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} does not exist.")

    # Load the images and image labels into training and validation set
    X_train, y_train, X_val, y_val = load_train_val_dataset(dir, classes)

    # Combine the training/validation images and image labels
    # Also create batches of size 32
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
    val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)

    train_data = train_data.map(lambda x, y: (x, tf.cast(y, tf.float32)))
    val_data = val_data.map(lambda x, y: (x, tf.cast(y, tf.float32)))

    if model_type == 'ResNet50':
        # Load pre-trained ResNet50 model without the classification layers
        base_model = ResNet50(weights='imagenet', include_top=False)

    elif model_type == 'ResNet101':
        # Load pre-trained ResNet101 model without the classification layers
        base_model = ResNet101(weights='imagenet', include_top=False)

    elif model_type == 'ResNet152':
        # Load pre-trained ResNet152 model without the classification layers
        base_model = ResNet152(weights='imagenet', include_top=False)

    else:
        raise ValueError(f"Invalid model_type: {model_type}. Expected one of: 'ResNet50', 'ResNet101', 'ResNet152'.")

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

    # Early stopping callback
    callback = EarlyStopping(restore_best_weights = True, patience=3)

    # Fit the model
    history = model.fit(train_data, epochs= 100, validation_data=val_data, callbacks = [callback])

    def plot_history(history):
        # Plot training & validation loss values
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot training & validation metric values
        plt.subplot(1, 2, 2)
        plt.plot(history.history['fbeta_score'], label='Training F-Score')
        plt.plot(history.history['val_fbeta_score'], label='Validation F-Score')
        plt.title('Model F-Score')
        plt.xlabel('Epoch')
        plt.ylabel('F-Score')
        plt.legend()

        plt.tight_layout()
        plt.show()

    # Call the function to plot
    plot_history(history)

    # Get the path where to store the model
    model_path = os.path.join(model_dir, model_name)

    # Store the model
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

except Exception as e:
    print(f"An error occurred: {e}")