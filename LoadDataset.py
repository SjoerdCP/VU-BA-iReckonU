# Import packages and functions
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

def read_filepaths(data_dir, classes = ['Blurred', 'Sharp']):
    """
    read_filepaths(img_dir, classes) is a function that takes as input:
      - data_dir: the directory where the data is stored. This directory should include folders that contain the images
      - classes: a list that contains the classes. These classes should be folders in data_dir that contain images in jpg format
    
    The function gives as output:
      - A pandas DataFrame that contains:
          - 'file_path': the path to the corresponding image
          - 'label': the label that corresponds to image
    """

    # Create arrays for image location and label
    file_paths = []
    labels = []

    # Loop over all classes
    for class_ in classes:
        try:
            # Get the path of the class folder that contains the images
            class_path = os.path.join(data_dir, class_)

            # Check if the class path exists
            if not os.path.isdir(class_path):
                raise FileNotFoundError(f"The directory '{class_path}' does not exist.")

            # Loop over all files in the class folder
            for file_name in os.listdir(class_path):
                
                # If the file is a .jpg (image) file store image location and label
                if file_name[-4:].lower() == '.jpg':
                    file_paths.append(os.path.join(class_path, file_name))
                    labels.append(class_)

        except FileNotFoundError as e:
            print(f"Error: {e}")

    # Create a dataframe containing image locations and labels
    df = pd.DataFrame({'file_path': file_paths, 'label': labels})

    # Return dataframe
    return df

def load_images(df, img_size = 256):
    """
    load_images(df, img_size) takes as input:
        - df: a pandas DataFrame that contains:
            - 'file_path': the path to a corresponding image
            - 'label': a label to a corresponding image
        - img_size: the size that the images should be loaded to. The size of the images will be (img_size x img_size x 3)

    The function gives as output:
        - a numpy array that contains the images of size (img_size x img_size x 3) that are specified by 'file_path' in df
        - a numpy array that contains that contains the labels of the images specified by 'file_path' in df
    """

    # Create arrays to store images and image labels
    images = []
    labels = []

    # Loop over all rows (images) in the dataframe to load them
    for index, row in df.iterrows():
        try:
            # Load the image
            img = load_img(row['file_path'], target_size=(img_size, img_size))
            img_array = img_to_array(img)

            # Store the image and image label
            images.append(img_array)
            labels.append(row['label'])

        except FileNotFoundError:
            print(f"Warning: Image file '{row['file_path']}' not found. Skipping this image.")

        except Exception as e:
            print(f"Error loading image '{row['file_path']}': {e}")

    # Return the images and image labels
    return np.array(images), np.array(labels)

def load_train_val_dataset(data_dir, classes = ['Blurred', 'Sharp'], val_size = 0.2):
    """
    load_train_val_dataset(data_dir, classes) takes as input:
        - data_dir: the directory where the data is stored. This directory should include folders that contain the images
        - classes: a list that contains two classes. The first class is the positive class and the second class is the negative class.
                   These classes should be folders in data_dir that contain images in .jpg format
        - val_size: the size of the validation set. This should be a float in the range of (0.0, 1.0) and indicates the proportion of the validation set

    The function gives as output:
        - X_train: a numpy array that contains the images that are used for training
        - y_train: a numpy array that contains the labels of the images used for training
        - X_val: a numpy array that contains the images that are used for validating
        - y_val: a numpy array that contains the labels of the images used for validating

    During the split into validation and training the majority class is undersampled and the classes are stratified
    """

    # Create a dataframe with image locations and labels
    df = read_filepaths(data_dir, classes)

    # Count the occurrences of each class
    class_counts = df['label'].value_counts()

    # Determine the majority and minority classes
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()

    # Separate majority and minority classes
    df_majority = df[df['label'] == majority_class]
    df_minority = df[df['label'] == minority_class]

    # Undersample majority class
    df_majority_undersampled = resample(df_majority, replace = False, n_samples = len(df_minority))

    # Combine both into dataframe into a single dataframe
    df_undersampled = pd.concat([df_minority, df_majority_undersampled])

    # Split the data into training and validation set, while stratifying the classes
    train_df, val_df = train_test_split(df_undersampled	, test_size = val_size, stratify = df_undersampled['label'])

    # Load training images and labels
    X_train, y_train = load_images(train_df)

    # Load validation images and labels
    X_val, y_val = load_images(val_df)

    # Convert labels to binary (0 and 1)
    label_to_index = {classes[0]: 1, classes[1]: 0}
    y_train = np.array([label_to_index[label] for label in y_train])
    y_val = np.array([label_to_index[label] for label in y_val])

    # Change the shape of the labels
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)

    # Return the training and validation images and labels
    return X_train, y_train, X_val, y_val

def load_public_dataset(excel_file, img_dir):
    """
    load_public_testset(excel_file, directory) takes as input:
        - excel_file: an excel file that only contains the following columns:
            - 'Image Name': this should contain the file name of the image
            - 'Blur Label': this should contain the label of the image:
                - (-1): sharp
                - (1): blurred
        - img_dir: the directory where the images, in .jpg format, are stored

    The function gives as output:
        - X_test: a numpy array that contains the images that are used for testing
        - y_test: a numpy array that contains the labels of the images used for testing
    """
    try:
        # Check if the Excel file exists
        if not os.path.isfile(excel_file):
            raise FileNotFoundError(f"The Excel file '{excel_file}' does not exist.")

        # Check if the image directory exists
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"The image directory '{img_dir}' does not exist.")
        
        # Read the Excel file into a dataframe
        df = pd.read_excel(excel_file)

        # Define file extension
        file_extension = '.jpg'

        # Function to convert filename to file path
        def filename_to_filepath(filename):
            return os.path.join(img_dir, filename + file_extension)

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
    
    except FileNotFoundError as e:
        print(f"Error: {e}")

def load_sewer_dataset(data_dir, classes = ['Blurred', 'Sharp']):
    """
    load_sewer_dataset(data_dir, classes) takes as input:
        - data_dir: the directory where the data is stored. This directory should include folders that contain the images
        - classes: a list that contains two classes. The first class is the positive class and the second class is the negative class.
                   These classes should be folders in data_dir that contain images in .jpg format

    The function gives as output:
        - X_test: a numpy array that contains the images that are used for testing
        - y_test: a numpy array that contains the labels of the images used for testing
    """

    # Create a dataframe with image locations and labels
    df = read_filepaths(data_dir, classes)

    # Load images and images labels
    X_test, y_test = load_images(df)

    # Change the image labels to binary format (Sharp: 0, Blurry: 1)
    label_to_index = {classes[0]: 1, classes[1]: 0}
    y_test = np.array([label_to_index[label] for label in y_test])
    
    # Change the shape of the image labels
    y_test = y_test.reshape(-1, 1)

    # Return the images and image labels
    return X_test, y_test

def augment_and_save_images(image_dir, save_dir, augment_count=5):
    """
    augment_and_save_images(image_dir, save_dir, augment_count) takes as input:
        - image_dir the directory where the original images are stored
        - save_dir: the directory where the augmented images should be stored
        - augment_count: the number of augmentations per image

    As output the function stores augment_count number of augmented images in the directory save_dir
    """

    try:
        # Check if the image directory exists
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"The directory '{image_dir}' does not exist.")

        # Create the save_dir if it does not exist already
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Define the augmentation parameters
        datagen = ImageDataGenerator(
            vertical_flip = True,
            horizontal_flip=True,
        )
        
        # List all images in the directory
        image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.JPG'))]
        
        # Loop over all images
        for image_file in image_files:

            # Load the image
            img = load_img(image_file)
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            
            # Create an iterator for augmentations
            it = datagen.flow(x, batch_size=1)
            
            # Generate and save augmented images
            base_name = os.path.basename(image_file).split('.')[0]

            for i in range(augment_count):
                augmented_image = next(it)[0].astype(np.uint8)

                save_path = os.path.join(save_dir, f"{base_name}_aug_{i+1}.jpg")
                array_to_img(augmented_image).save(save_path)

    except FileNotFoundError as e:
        print(f"Error: {e}")