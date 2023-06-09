import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import sklearn; print("SkLearn", sklearn.__version__)

import os
import pandas as pd
import numpy as np
from datetime import date
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from joblib import dump

# inputs
TRAIN_FP = "../data/titanic_train_preprocessed.csv"
TARGET = "Survived"
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
TEST_PROP = 0.2

# outputs
# BOOSTING_MODEL_SAVE_FP = f"../model/titanic_boosting_model.csv"
# RANDOM_FOREST_MODEL_SAVE_FP = f"../model/titanic_random_forest_model.csv"

def train():

    MODEL_DIR = os.environ["MODEL_DIR"]
    MODEL_FILE_GB = os.environ["MODEL_FILE_GB"]
    MODEL_FILE_RF = os.environ["MODEL_FILE_RF"]
    MODEL_PATH_GB = os.path.join(MODEL_DIR, MODEL_FILE_GB)
    MODEL_PATH_RF = os.path.join(MODEL_DIR, MODEL_FILE_RF)

    print("***** loading and preprocessing data *****")
    df = pd.read_csv(TRAIN_FP, usecols=FEATURES + [TARGET])
    df = pd.get_dummies(df)
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    print(f"training data shape: {X.shape}, {y.shape}")
    
    print("***** fitting and saving gb and rf models *****")
    gb_clf = GradientBoostingClassifier().fit(X, y)
    dump(gb_clf, MODEL_PATH_GB)

    rf_clf = RandomForestClassifier().fit(X, y)
    dump(rf_clf, MODEL_PATH_RF)

    print("***** training complete *****")

    
if __name__ == "__main__":
    train()