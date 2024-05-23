import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import metrics
from tensorflow.keras import backend as K

def fbeta_score(y_true, y_pred, beta=0.5):
    # Ensure y_true and y_pred are of type float32
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    # Round predictions to get binary values
    y_pred = K.round(y_pred)
    
    # Compute the number of true positives
    tp = K.sum(y_true * y_pred, axis=0)
    
    # Compute the number of predicted positives
    pred_positives = K.sum(y_pred, axis=0)
    
    # Compute the number of actual positives
    actual_positives = K.sum(y_true, axis=0)
    
    # Compute precision and recall
    precision = tp / (pred_positives + K.epsilon())
    recall = tp / (actual_positives + K.epsilon())
    
    # Compute the F0.5 score
    beta_squared = beta ** 2
    fbeta = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())
    
    return fbeta


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
              metrics=[fbeta_score])

    return model

def fit_model(model, epochs, X_train, y_train, X_val, y_val):
    history = model.fit(X_train, y_train, epochs=epochs,
                        validation_data=(X_val, y_val))

    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f'Validation loss: {val_loss}, Validation accuracy: {val_acc}')

    return history

