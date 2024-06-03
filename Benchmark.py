from LoadDataset import load_dataset
from LoadPublicDataset import load_public_dataset
from PreprocessData import preprocess_data
from EvaluateModel import evaluate_model
from LaplaceModel import LaplaceModel
from matplotlib.patches import Rectangle
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, fbeta_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import tensorflow as tf 
tf.keras.utils.set_random_seed(12345)

blur_folder = './PublicDataset/Train/Naturally-Blurred'
sharp_folder = './PublicDataset/Train/Undistorted'

#X_train, X_val, y_train, y_val = load_public_dataset(sharp_folder, blur_folder)
X_train, X_val, y_train, y_val = load_public_dataset('Sharp', 'Blurry')

print('Loaded Dataset')

X_train = preprocess_data(X_train)
X_val = preprocess_data(X_val)

print('Preproccesed Dataset')

model = LaplaceModel()
model.fit(X_train, y_train)
#model.set_threshold(0.05)

# y_pred = model.predict(X_train)
# conf_matrix = confusion_matrix(y_train, y_pred)
# print(f'Train Matrix:\n {conf_matrix}')
# print(fbeta_score(y_train, y_pred, beta=1.5))

y_pred = model.predict(X_val)

# Calculate the required metrics
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
#roc_auc = roc_auc_score(y_val, y_pred_prob)  # Use probabilities for ROC AUC
conf_matrix = confusion_matrix(y_val, y_pred, labels = [0, 1])
fbeta = fbeta_score(y_val, y_pred, beta = 0.5)

# Printing the metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'F0.5 Score: {fbeta}')
#print(f'ROC-AUC Score: {roc_auc}')
print(f'Confusion Matrix:\n{conf_matrix}')

print(model.threshold)
print(y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Sharp', 'Blurred'])
disp.plot(cmap = 'Reds', colorbar = False, text_kw = {'fontsize': 12})

ax = plt.gca()

# Optionally, you can also increase the font size of the axis labels
ax.set_xlabel('Predicted label', fontsize=14)
ax.set_ylabel('True label', fontsize=14)
ax.set_xticklabels(['Sharp', 'Blurred'], fontsize=12)
ax.set_yticklabels(['Sharp', 'Blurred'], fontsize=12)

plt.show()