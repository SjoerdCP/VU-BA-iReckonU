from LoadDataset import load_public_dataset, load_sewer_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score, confusion_matrix, ConfusionMatrixDisplay
import pickle
import os
import time
import numpy as np
import matplotlib.pyplot as plt

def calculate_metrics(y_true, y_pred, show_matrix = True):
    """
    calculate_metrics(y_true, y_pred) takes as input:
        - y_true: a numpy array containing the true labels of the images
        - y_pred: a numpy array containing the predicted labels of the images
        - show_matrix: a boolean indicating if the confusion matrix should be shown

    The function gives as output:
        - A dictionary containing the following metrics and their value:
            - 'F_{0.5}-score'
            - 'Accuracy'
            - 'Precision'
            - 'Recall'
        - Additionally it shows a confusion matrix if show_matrix is True
    """
    # Create the dictionary
    metrics_dict = {}

    # Calculate the metrics and store them
    metrics_dict['F_{0.5}-score'] = fbeta_score(y_true, y_pred, beta = 0.5)
    metrics_dict['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics_dict['Precision'] = precision_score(y_true, y_pred)
    metrics_dict['Recall'] = recall_score(y_true, y_pred)

    # If show_matrix is True plot a confusion matrix
    if show_matrix:
        conf_matrix = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Sharp', 'Blurred'])
        disp.plot(cmap = 'Reds', colorbar = False, text_kw = {'fontsize': 12})

        ax = plt.gca()
        ax.set_xlabel('Predicted label', fontsize=14)
        ax.set_ylabel('True label', fontsize=14)
        ax.set_xticklabels(['Sharp', 'Blurred'], fontsize=12)
        ax.set_yticklabels(['Sharp', 'Blurred'], fontsize=12)

        plt.show()
    
    # Return the metrics dictionary
    return metrics_dict

def evaluate_model_sewer(data_dir, model_dir, model_name, classes = ['Blurred', 'Sharp'], show_matrix = True):
    """
    evaluate_model_sewer(data_dir, model_dir, model_name, classes) takes as input:
        - data_dir: the directory where the data is stored. This directory should include folders that contain the images
        - model_dir: the directory where the model is stored
        - model_name: the file name of the model that is used to do the classification
        - classes: a list that contains two classes. The first class is the positive class and the second class is the negative class.
                   These classes should be folders in data_dir that contain images in .jpg format
        - show_matrix: a boolean indicating if the confusion matrix should be shown

    The function gives as output:
        - The elapsed time that it took to load the images and classify them
        - A dictionary containing the following metrics and their value:
            - 'F_{0.5}-score'
            - 'Accuracy'
            - 'Precision'
            - 'Recall'
        - Additionally it shows a confusion matrix if show_matrix is True
    """

    # Get path to the model
    model_path = os.path.join(model_dir, model_name)

    # Open the model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    # Start timer
    start_time = time.time()

    # Load images
    X_test, y_test = load_sewer_dataset(data_dir, classes)

    # Classify images
    y_pred = np.where(model.predict(X_test) > 0.5, 1, 0)

    # End timer
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    # Calculate the metrics
    metrics_dict = calculate_metrics(y_test, y_pred, show_matrix)

    # Print the metrics
    print(metrics_dict)

    # Return the elapsed time and metrics
    return elapsed_time, metrics_dict

def evaluate_model_public(excel_file, img_dir, model_dir, model_name, show_matrix = True):
    """
    evaluate_model_public(data_dir, classes, model_dir, model_name) takes as input:
        - excel_file: an excel file that only contains the following columns:
            - 'Image Name': this should contain the file name of the image
            - 'Blur Label': this should contain the label of the image:
                - (-1): sharp
                - (1): blurred
        - img_dir: the directory where the images, in .jpg format, are stored
        - model_dir: the directory where the model is stored
        - model_name: the file name of the model that is used to do the classification
        - show_matrix: a boolean indicating if the confusion matrix should be shown

    The function gives as output:
        - The elapsed time that it took to load the images and classify them
        - A dictionary containing the following metrics and their value:
            - 'F_{0.5}-score'
            - 'Accuracy'
            - 'Precision'
            - 'Recall'
        - Additionally it shows a confusion matrix if show_matrix is True
    """

    # Get path to the model
    model_path = os.path.join(model_dir, model_name)

    # Open the model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    # Start timer
    start_time = time.time()

    # Load images
    X_test, y_test = load_public_dataset(excel_file, img_dir)

    # Classify images
    y_pred = np.where(model.predict(X_test) > 0.5, 1, 0)

    # End timer
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    # Calculate the metrics
    metrics_dict = calculate_metrics(y_test, y_pred, show_matrix)

    # Print the metrics
    print(metrics_dict)

    # Return the elapsed time and metrics
    return elapsed_time, metrics_dict

evaluate_model_sewer('SewerImgs/MajorityNoText', 'Models', 'ResNet50.pkl')
# evaluate_model_sewer('SewerImgs/MajorityNoText', 'Models', 'AugmentedMaxResNet152.pkl')
# evaluate_model_public('PublicDataset/Evaluate/NaturalBlurSet.xlsx', 'PublicDataset/Evaluate/NaturalBlurSet', 'Models', 'AugmentedMaxResNet152.pkl') 