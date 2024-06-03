import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_dataset_new(data_dir, categories):

    # Create a DataFrame with file paths and labels
    file_paths = []
    labels = []

    for category in categories:
        category_path = os.path.join(data_dir, category)
        for file_name in os.listdir(category_path):
            if file_name[-4:] == '.jpg':
                file_paths.append(os.path.join(category, file_name))
                print(file_paths)
                labels.append(category)

    df = pd.DataFrame({'file_path': file_paths, 'label': labels})

    # Undersample the majority class in the training data
    # Separate majority and minority classes
    df_majority = df[df['label'] == categories[1]]
    df_minority = df[df['label'] == categories[0]]

    # Undersample majority class
    df_majority_undersampled = resample(df_majority,
                                        replace=False,     # Sample without replacement
                                        n_samples=len(df_minority))    # Match number of samples in minority class

    # Combine minority class with undersampled majority class
    df_undersampled = pd.concat([df_minority, df_majority_undersampled])

    # Perform stratified splitting
    train_df, val_df = train_test_split(df_undersampled	, test_size=0.2, stratify=df_undersampled['label'])

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

    # Load training images and labels
    X_train, y_train = load_images(train_df)

    # # Load validation images and labels
    X_val, y_val = load_images(val_df)

    # # Convert labels to binary (0 and 1)
    label_to_index = {categories[0]: 1, categories[1]: 0}
    y_train = np.array([label_to_index[label] for label in y_train])
    y_val = np.array([label_to_index[label] for label in y_val])

    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)

    return X_train, y_train, X_val, y_val, 
