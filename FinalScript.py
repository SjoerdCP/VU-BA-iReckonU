import pickle
import pandas as pd
import numpy as np
import os
import tensorflow as tf

def classify_images(img_dir, out_path, model_dir, model_name):
    """
    classify_images(img_dir, out_path, model_dir, model_name) is a function that takes as input:
      - img_dir: the directory where the images that need to be classified are stored
      - out_path: the path (directory + csv file name) where the output csv file needs to be stored
      - model_dir: the directory where the model is stored
      - model_name: the file name of the model that is used to do the classification
    
    As output the function creates a csv file that is stored at out_path. This csv file contains:
      - img: the file name of the image
      - predicted_label: the label that is predicted by the model:
          - 0: indicates that the images is sharp
          - 1: indicates that the image is blurred
    """

    try:
        # Check if img_dir, model_dir and model_name exist
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"The image directory '{img_dir}' does not exist.")
        
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"The model directory '{model_dir}' does not exist.")
        
        model_path = os.path.join(model_dir, model_name)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"The model file '{model_path}' does not exist.")
        
        # Set the size of the images
        img_size = 256

        # Create arrays to store the image names and the images
        img_names = np.array(os.listdir(img_dir), dtype=object)
        imgs = np.zeros((len(img_names), img_size, img_size, 3))

        # Loop over all images to store them in the imgs array
        for i, img_name in enumerate(img_names):
            img_path = os.path.join(img_dir, img_name)

            img = tf.keras.preprocessing.image.load_img(img_path, img_size, target_size=(img_size, img_size))
            imgs[i] = tf.keras.preprocessing.image.img_to_array(img)

        # Create the path to the model
        model_path = os.path.join(model_dir, model_name)

        # Open the model
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        
        # Classify the images as either sharp (0) or blurred (1)
        predicted_labels = np.where(model.predict(imgs) > 0.5, 1, 0).flatten()

        # Create a dataframe that contains the images names and predicted labels 
        df = pd.DataFrame({'img': img_names, 'predicted_labels': predicted_labels})

        # Create the CSV containing the image names and the predicted labels
        df.to_csv(out_path, index = False)

    except FileNotFoundError as e:
        print(f"Error: {e}")

from LoadDataset import read_filepaths, load_images

df = read_filepaths('SewerImgs/Lex')
imgs, labs = load_images(df)

# img_dir = 'SewerImgs/Original'
# out_name = 'ClassTest.csv'
# model_dir = 'Models'
# model_name = 'AugmentedMaxResNet152.pkl'

# classify_images(img_dir, out_name, model_dir, model_name)