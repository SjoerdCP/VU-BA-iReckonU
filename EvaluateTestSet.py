# Import packages and functions
from LoadDataset import load_public_testset, load_sewer_dataset
from EvaluateModel import evaluate_model
import pickle
import time

# Set parameters
public_testset = False
cnn = False

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
    data_dir = 'SewerImgs/Majority'
    classes = ['Blurred', 'Sharp']

    # data_dir = 'SewerImgs/MajorityNoText'
    # classes = ['Blurred', 'Sharp']

    # Load images and labels
    X_test, y_test = load_sewer_dataset(data_dir, classes)

# Load model
model = 'AugmentedMaxResNet50' if cnn else 'public_laplacian_gaussian'
model = 'Models/' + model + '.pkl'

with open(model, 'rb') as file:
    model = pickle.load(file)

start_time = time.time()
y_pred = model.predict(X_test)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
# Evaluate model
evaluate_model(model, X_test, y_test)