import os
import pandas as pd
import shutil

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
                file_paths.append(file_name)
                labels.append(class_)

    # Create a dataframe containing image locations and labels
    df = pd.DataFrame({'file_path': file_paths, 'label': labels})

    # Return dataframe
    return df

classes = ['Blurred', 'Sharp']
lex = read_filepaths('SewerImgs/Lex', classes)
joeri = read_filepaths('SewerImgs/Joeri', classes)
sjoerd = read_filepaths('SewerImgs/Sjoerd', classes)
kay = read_filepaths('SewerImgs/Kay', classes)
joshua = read_filepaths('SewerImgs/Joshua', classes)

merged_df = lex
for df in [joeri, sjoerd, kay, joshua]:
    merged_df = pd.merge(merged_df, df, on='file_path', how='outer', suffixes=('', '_dup'))

merged_df.columns = ['file_name', 'lex', 'joeri', 'sjoerd', 'kay', 'joshua']

# Function to determine the majority vote
def majority_vote(labels):
    return labels.mode()[0]

# Exclude the first column and apply the majority vote function
merged_df['final_label'] = merged_df.iloc[:, 1:].apply(majority_vote, axis=1)

source_dir = 'SewerImgs/Original'
blurred_dir = 'SewerImgs/Majority/Blurred'
sharp_dir = 'SewerImgs/Majority/Sharp'

for index, row in merged_df.iterrows():
    src_path = os.path.join(source_dir, row['file_name'])
    if row['final_label'] == 'Blurred':
        dst_path = os.path.join(blurred_dir, row['file_name'])
    elif row['final_label'] == 'Sharp':
        dst_path = os.path.join(sharp_dir, row['file_name'])
    else:
        continue  # Handle other cases if needed

    # Copy the file
    shutil.copy2(src_path, dst_path)