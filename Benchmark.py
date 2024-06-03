from LoadDataset import load_dataset_new
from PreprocessData import preprocess_data
from EvaluateModel import evaluate_model
from LaplaceModel import LaplaceModel
import tensorflow as tf 

tf.keras.utils.set_random_seed(12345)

# data_dir = ''
# categories = ['Blurry', 'Sharp']

data_dir = 'PublicDataset/Train'
categories = ['Naturally-Blurred', 'Undistorted']

X_train, y_train, X_val, y_val = load_dataset_new(data_dir, categories)
print('Loaded Dataset')

X_train = preprocess_data(X_train)
X_val = preprocess_data(X_val)

print('Preproccesed Dataset')

model = LaplaceModel()
model.fit(X_train, y_train)

evaluate_model(model, X_val, y_val)
