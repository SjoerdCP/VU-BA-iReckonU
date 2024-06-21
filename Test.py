from LoadDataset import load_public_testset, load_sewer_dataset
import pickle
from tensorflow.keras.preprocessing.image import array_to_img
import os
import numpy as np

save_dir = 'Results'

model = 'AugmentedMaxResNet152'
model = 'Models/' + model + '.pkl'

with open(model, 'rb') as file:
    model = pickle.load(file)

data_dir = 'SewerNoText'
classes = ['Blurred', 'Sharp']

X_test, y_test = load_sewer_dataset(data_dir, classes)

y_pred_prob = model.predict(X_test)
y_pred = np.where(y_pred_prob > 0.5, 1, 0)

# True Positives: both y_true and y_pred are 1
TP_indices = [i for i, (yt, yp) in enumerate(zip(y_test, y_pred)) if yt == 1 and yp == 1]

# True Negatives: both y_true and y_pred are 0
TN_indices = [i for i, (yt, yp) in enumerate(zip(y_test, y_pred)) if yt == 0 and yp == 0]

# False Positives: y_true is 0 but y_pred is 1
FP_indices = [i for i, (yt, yp) in enumerate(zip(y_test, y_pred)) if yt == 0 and yp == 1]

# False Negatives: y_true is 1 but y_pred is 0
FN_indices = [i for i, (yt, yp) in enumerate(zip(y_test, y_pred)) if yt == 1 and yp == 0]

for i, img in enumerate(X_test):
    if i in TP_indices:
        folder = 'TP'
    elif i in TN_indices:
        folder = 'TN'
    elif i in FP_indices:
        folder = 'FP'
    else: folder = 'FN'

    save_path = os.path.join(save_dir, folder, f'img{i}.jpg')

    array_to_img(img).save(save_path)

print(y_pred)