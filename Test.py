import pandas as pd
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from LaplaceModel import LaplaceModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, fbeta_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from EvaluateModel import evaluate_model
from PreprocessData import preprocess_data

# # Replace 'your_excel_file.xlsx' with the path to your Excel file
# file_path = './PublicDataset/Evaluate/NaturalBlurSet.xlsx'

# # Read the Excel file into a DataFrame
# df = pd.read_excel(file_path)

# # Define the directory path and file extension
# directory = 'PublicDataset/Evaluate/NaturalBlurSet'
# file_extension = '.jpg'

# # Function to convert filename to file path
# def filename_to_filepath(filename):
#     return os.path.join(directory, filename + file_extension)

# # Apply the function to the 'filename' column and create a new 'file_path' column
# df['Image Name'] = df['Image Name'].apply(filename_to_filepath)
# df['Blur Label'] = df['Blur Label'].replace(-1, 0)

# # Parameters
# img_height = 256
# img_width = 256

# def load_images(df):
#     images = []
#     labels = []
#     for index, row in df.iterrows():
#         img = load_img(row['Image Name'], target_size=(img_height, img_width))
#         img_array = img_to_array(img)
#         images.append(img_array)
#         labels.append(row['Blur Label'])
#     return np.array(images), np.array(labels)

# Define paths
categories = ['Blurry', 'Sharp']

# Create a DataFrame with file paths and labels
file_paths = []
labels = []

for category in categories:
    for file_name in os.listdir(category):
        if file_name[-4:] == '.jpg':
            file_paths.append(os.path.join(category, file_name))
            labels.append(category)

df = pd.DataFrame({'file_path': file_paths, 'label': labels})

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

X_test, y_test = load_images(df)

label_to_index = {'Blurry': 1, 'Sharp': 0}
y_test = np.array([label_to_index[label] for label in y_test])

y_test = y_test.reshape(-1, 1)

model = load_model('TestResNet.keras')

evaluate_model(model, X_test, y_test)

# model = LaplaceModel()
# model.set_threshold(0.01890320665502987)

# X_test = preprocess_data(X_test)
# y_pred = model.predict(X_test)

# # Calculate the required metrics
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# #roc_auc = roc_auc_score(y_test, y_pred_prob)  # Use probabilities for ROC AUC
# conf_matrix = confusion_matrix(y_test, y_pred, labels = [0, 1])
# fbeta = fbeta_score(y_test, y_pred, beta = 0.5)

# # Printing the metrics
# print(f'Accuracy: {accuracy}')
# print(f'Precision: {precision}')
# print(f'Recall: {recall}')
# print(f'F1 Score: {f1}')
# print(f'F0.5 Score: {fbeta}')
# #print(f'ROC-AUC Score: {roc_auc}')
# print(f'Confusion Matrix:\n{conf_matrix}')

# print(model.threshold)
# print(y_pred)

# disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Sharp', 'Blurred'])
# disp.plot(cmap = 'Reds', colorbar = False, text_kw = {'fontsize': 12})

# ax = plt.gca()

# # Optionally, you can also increase the font size of the axis labels
# ax.set_xlabel('Predicted label', fontsize=14)
# ax.set_ylabel('True label', fontsize=14)
# ax.set_xticklabels(['Sharp', 'Blurred'], fontsize=12)
# ax.set_yticklabels(['Sharp', 'Blurred'], fontsize=12)

# plt.show()