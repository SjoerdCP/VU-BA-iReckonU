import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define a simple CNN model
model = Sequential([
    # Assuming images are 256x256 and grayscale
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.summary()  # To see the model architecture


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Reshape for the model (adding channel dimension)
X_train = X_train.reshape((-1, 256, 256, 1))
X_val = X_val.reshape((-1, 256, 256, 1))

# Normalize the data
X_train = X_train / 255.0
X_val = X_val / 255.0


history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_val, y_val))


val_loss, val_acc = model.evaluate(X_val, y_val)
print(f'Validation loss: {val_loss}, Validation accuracy: {val_acc}')


