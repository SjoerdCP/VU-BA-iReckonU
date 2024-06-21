import os
import numpy as np
import pandas as pd
from LoadDataset import read_filepaths, load_images

import matplotlib.pyplot as plt
import seaborn as sns

dir = 'Results'

folders = ['FN', 'FP', 'TN', 'TP']

df = read_filepaths(dir, folders)
df['correct'] = [True if lab.startswith('T') else False for lab in df['label']]

imgs, labels = load_images(df)

color_distributions = []
sizes = []

imgs = imgs.astype(np.int64)

def calculate_mean_var_intensity(image):
    mean_intensities = [np.mean(image[:, :, c]) for c in range(image.shape[2])]
    var_intensities = [np.var(image[:, :, c]) for c in range(image.shape[2])]
    return mean_intensities + var_intensities

# Calculate the mean intensities for all images
mean_intensities = [calculate_mean_var_intensity(img) for img in imgs]

# Create a DataFrame
df_color = pd.DataFrame(mean_intensities, columns=['Mean_R', 'Mean_G', 'Mean_B', 'Var_R', 'Var_G', 'Var_B'])

merged_df = pd.merge(df, df_color, left_index = True, right_index = True)

sns.histplot(data = merged_df, x = 'Mean_R', hue = 'correct', stat = 'density', common_norm = False)
plt.show()

sns.histplot(data = merged_df, x = 'Mean_G', hue = 'correct', stat = 'density', common_norm = False)
plt.show()

sns.histplot(data = merged_df, x = 'Mean_B', hue = 'correct', stat = 'density', common_norm = False)
plt.show()

sns.histplot(data = merged_df, x = 'Var_R', hue = 'correct', stat = 'density', common_norm = False)
plt.show()

sns.histplot(data = merged_df, x = 'Var_G', hue = 'correct', stat = 'density', common_norm = False)
plt.show()

sns.histplot(data = merged_df, x = 'Var_B', hue = 'correct', stat = 'density', common_norm = False)
plt.show()

# for img in imgs:
#     color_distribution = np.bincount(img.flatten(), minlength = 256)
#     color_distributions.append(color_distribution)

# color_distributions = np.array(color_distributions)

# print(color_distributions)



# # Calculate the mean color distribution across all images
# mean_color_distribution = np.mean(color_distributions, axis=0)

# # Plot a bar chart of the mean color distribution
# plt.bar(np.arange(256), mean_color_distribution)
# plt.title("Mean Color Distribution")
# plt.xlabel("Color Value")
# plt.ylabel("Number of Pixels")
# plt.show()