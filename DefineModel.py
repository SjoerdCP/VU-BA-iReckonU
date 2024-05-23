import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_model():
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
              metrics=[tf.keras.metrics.FBetaScore(beta = 0.5, threshold = 0.5)])

    return model

def fit_model(model, epochs, X_train, y_train, X_val, y_val):
    history = model.fit(X_train, y_train, epochs=epochs,
                        validation_data=(X_val, y_val))

    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f'Validation loss: {val_loss}, Validation accuracy: {val_acc}')

    return history

