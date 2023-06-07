import pandas as pd
import numpy as np
from datetime import date
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from joblib import dump

# inputs
TRAIN_FP = "../data/train.csv"
TARGET = "Survived"
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
TEST_PROP = 0.2

# outputs
BOOSTING_MODEL_SAVE_FP = f"../model/titanic_boosting_model.csv"
RANDOM_FOREST_MODEL_SAVE_FP = f"../model/titanic_random_forest_model.csv"

def train():

    print("***** loading and preprocessing data *****")
    df = pd.read_csv(TRAIN_FP, usecols=FEATURES + [TARGET])
    df = df.dropna(subset=["Embarked"]) # drop 2 rows with missing embarked
    df["Age"] = df["Age"].fillna(df["Age"].mean()) # fill missing age values with mean age (~30yrs)
    # print(df.shape)
    # print(df.head())
    df = pd.get_dummies(df)
    # print(f"dummified data:\n{df.head()}")
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    print(f"training data shape: {X.shape}, {y.shape}")
    
    print("***** fitting and saving gb and rf models *****")
    gb_clf = GradientBoostingClassifier().fit(X, y)
    dump(gb_clf, BOOSTING_MODEL_SAVE_FP)

    rf_clf = RandomForestClassifier().fit(X, y)
    dump(rf_clf, RANDOM_FOREST_MODEL_SAVE_FP)

    print("***** training complete *****")

    
if __name__ == "__main__":
    train()