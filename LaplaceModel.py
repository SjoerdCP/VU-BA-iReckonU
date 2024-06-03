import cv2
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, fbeta_score
from scipy.optimize import dual_annealing
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def laplacian_var(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

class LaplaceModel:
    def __init__(self):
        self.threshold = -1

    def fit(self, X, y):
        def accuracy_min(threshold, vars, y):
            y_pred = [1 if var < threshold else 0 for var in vars]
            return -fbeta_score(y, y_pred, beta=0.5)
        
        vars = [laplacian_var(x) for x in X]

        threshold = [0]
        res = dual_annealing(accuracy_min, bounds = list(zip([0], [1])), x0= threshold, args = (vars, y))
        self.threshold = res.x[0]

        print(res)
        print(accuracy_min(self.threshold, vars, y))

        # df = pd.DataFrame(zip(vars, y), columns = ['var', 'label'])
        # sns.histplot(data = df, x = 'var', hue = 'label')
        # plt.show()
    
    def predict(self, X_test):
        return np.array([1 if laplacian_var(img) < self.threshold else 0 for img in X_test])
    
    def set_threshold(self, new_threshold):
        self.threshold = new_threshold
