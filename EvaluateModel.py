# Import packages and functions
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix, ConfusionMatrixDisplay

# Function to evaluate the model
def evaluate_model(model, X_test, y_true):

    # Predict labels using the model
    y_pred_prob = model.predict(X_test)
    y_pred = np.where(y_pred_prob > 0.5, 1, 0)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    f05 = fbeta_score(y_true, y_pred, beta= 0.5)

    # Create confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Printing metrics
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'F0.5 Score: {f05}')

    # Print confusion matrix
    print(f'Confusion Matrix:\n{conf_matrix}')
    
    # Display confusion matrix as a plot
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Sharp', 'Blurred'])
    disp.plot(cmap = 'Reds', colorbar = False, text_kw = {'fontsize': 12})

    ax = plt.gca()
    ax.set_xlabel('Predicted label', fontsize=14)
    ax.set_ylabel('True label', fontsize=14)
    ax.set_xticklabels(['Sharp', 'Blurred'], fontsize=12)
    ax.set_yticklabels(['Sharp', 'Blurred'], fontsize=12)

    plt.show()