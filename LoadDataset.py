# Import packages and functions
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Function to create a dataframe that contains image locations and labels
def read_filepaths(data_dir, classes):

    # Create arrays for image location and label
    file_paths = []
    labels = []

    # Loop over all classes
    for class_ in classes:
        
        # Get the path of the class folder that contains the images
        class_path = os.path.join(data_dir, class_)

        # Loop over all files in the class folder
        for file_name in os.listdir(class_path):
            
            # If the file is a .jpg (image) file store image location and label
            if file_name[-4:].lower() == '.jpg':
                file_paths.append(os.path.join(class_path, file_name))
                labels.append(class_)

    # Create a dataframe containing image locations and labels
    df = pd.DataFrame({'file_path': file_paths, 'label': labels})

    # Return dataframe
    return df

# Function to load all images in a dataframe
def load_images(df, img_height = 256, img_width = 256):

    # Create arrays to store images and image labels
    images = []
    labels = []

    # Loop over all rows (images) in the dataframe to load them
    for index, row in df.iterrows():

        # Load the image
        img = load_img(row['file_path'], target_size=(img_height, img_width))
        img_array = img_to_array(img)

        # Store the image and image label
        images.append(img_array)
        labels.append(row['label'])

    # Return the images and image labels
    return np.array(images), np.array(labels)

# Function to load a dataset into a training and validation set
# It also undersamples the majority class and stratifies the classes
def load_dataset(data_dir, categories):

    # Create a dataframe with image locations and labels
    df = read_filepaths(data_dir, categories)

    # Separate majority and minority classes
    df_majority = df[df['label'] == categories[1]]
    df_minority = df[df['label'] == categories[0]]

    # Undersample majority class
    df_majority_undersampled = resample(df_majority, replace=False, n_samples=len(df_minority))

    # Combine both into dataframe into a single dataframe
    df_undersampled = pd.concat([df_minority, df_majority_undersampled])

    # Split the data into training and validation set, while stratifying the classes
    train_df, val_df = train_test_split(df_undersampled	, test_size=0.2, stratify=df_undersampled['label'])

    # Load training images and labels
    X_train, y_train = load_images(train_df)

    # Load validation images and labels
    X_val, y_val = load_images(val_df)

    # Convert labels to binary (0 and 1)
    label_to_index = {categories[0]: 1, categories[1]: 0}
    y_train = np.array([label_to_index[label] for label in y_train])
    y_val = np.array([label_to_index[label] for label in y_val])

    # Change the shape of the labels
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)

    # Return the training and validation images and labels
    return X_train, y_train, X_val, y_val

# Function to load the public testset
def load_public_testset(excel_file, directory):

    # Read the Excel file into a dataframe
    df = pd.read_excel(excel_file)

    # Define file extension
    file_extension = '.jpg'

    # Function to convert filename to file path
    def filename_to_filepath(filename):
        return os.path.join(directory, filename + file_extension)

    # Apply the filename_to_filepath function to change the filenames to filepaths in the dataframe
    df['Image Name'] = df['Image Name'].apply(filename_to_filepath)

    # Change the the -1 label of sharp images to 0
    df['Blur Label'] = df['Blur Label'].replace(-1, 0)

    # Change the column names
    df.columns = ['file_path', 'label']

    # Load the images and image labels
    X_test, y_test = load_images(df)

    # Change the shape of the image labels
    y_test = y_test.reshape(-1, 1)

    # Return the images and image labels
    return X_test, y_test

# Function to load the complete labeled sewer dataset as test set
def load_sewer_dataset(data_dir, categories):

    # Create a dataframe with image locations and labels
    df = read_filepaths(data_dir, categories)

    # Load images and images labels
    X_test, y_test = load_images(df)

    # Change the image labels to binary format (Sharp: 0, Blurry: 1)
    label_to_index = {categories[0]: 1, categories[1]: 0}
    y_test = np.array([label_to_index[label] for label in y_test])
    
    # Change the shape of the image labels
    y_test = y_test.reshape(-1, 1)

    # Return the images and image labels
    return X_test, y_test

