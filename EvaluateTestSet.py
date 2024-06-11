# Import packages and functions
from LoadDataset import load_public_testset, load_sewer_dataset
from EvaluateModel import evaluate_model
import pickle

# Set parameters
public_testset = False
cnn = True

# Load test data
if public_testset:

    # Set location to excel file
    excel_file = './PublicDataset/Evaluate/NaturalBlurSet.xlsx'

    # Set directory of images
    directory = 'PublicDataset/Evaluate/NaturalBlurSet'

    # Load images and labels
    X_test, y_test = load_public_testset(excel_file, directory)

else:

    # Set directory and classes
    data_dir = 'SewerNoText'
    classes = ['Blurred', 'Sharp']

    # data_dir = ''
    # classes = ['Blurry', 'Sharp']

    # Load images and labels
    X_test, y_test = load_sewer_dataset(data_dir, classes)

# Load model
model = 'AugmentedMaxResNet152' if cnn else 'public_laplacian'
model = 'Models/' + model + '.pkl'

with open(model, 'rb') as file:
    model = pickle.load(file)

# Evaluate model
evaluate_model(model, X_test, y_test)