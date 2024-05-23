import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def evaluate_model(model, X_test, y_true):
    # Predict classes or probabilities depending on the needs for metrics
    y_pred_prob = model.predict(X_test)
    y_pred = np.where(y_pred_prob > 0.5, 1, 0)  # Convert probabilities to binary output

    # Calculate the required metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_prob)  # Use probabilities for ROC AUC
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Printing the metrics
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'ROC-AUC Score: {roc_auc}')
    print(f'Confusion Matrix:\n{conf_matrix}')

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Sharp', 'Blurred'])
    disp.plot(cmap = 'Reds', colorbar = False, text_kw = {'fontsize': 12})

    ax = plt.gca()

    # Optionally, you can also increase the font size of the axis labels
    ax.set_xlabel('Predicted label', fontsize=14)
    ax.set_ylabel('True label', fontsize=14)
    ax.set_xticklabels(['Sharp', 'Blurred'], fontsize=12)
    ax.set_yticklabels(['Sharp', 'Blurred'], fontsize=12)

    plt.show()
