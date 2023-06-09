import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import sklearn; print("SkLearn", sklearn.__version__)

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from joblib import load

# inputs
TEST_FP = "../data/titanic_test_preprocessed.csv"
TARGET = "Survived"

def inference():

    MODEL_DIR = os.environ["MODEL_DIR"]
    MODEL_FILE_GB = os.environ["MODEL_FILE_GB"]
    MODEL_FILE_RF = os.environ["MODEL_FILE_RF"]
    MODEL_PATH_GB = os.path.join(MODEL_DIR, MODEL_FILE_GB)
    MODEL_PATH_RF = os.path.join(MODEL_DIR, MODEL_FILE_RF)

    df = pd.read_csv(TEST_FP)
        
    X_test = df.drop(TARGET, axis = 1)
    y_test = df[TARGET].values
   
    print("Shape of the test data")
    print(X_test.shape)
    print(y_test.shape)
        
    # Run model
    gb_clf = load(MODEL_PATH_GB)
    print("gradient boosting score and classification:")
    print(gb_clf.score(X_test, y_test))
    print(gb_clf.predict(X_test))
        
    # Run model
    rf_clf = load(MODEL_PATH_RF)
    print("random forest score and classification:")
    print(rf_clf.score(X_test, y_test))
    print(rf_clf.predict(X_test))
    
if __name__ == '__main__':
    inference()