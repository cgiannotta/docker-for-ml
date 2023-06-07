import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from joblib import dump

# inputs
TEST_FP = "../data/test.csv"
TARGET = "Survived"

def inference():

    df = pd.read_csv(TEST_FP)
        
    y_test = data_test['# Letter'].values
    X_test = data_test.drop(data_test.loc[:, 'Line':'# Letter'].columns, axis = 1)
   
    print("Shape of the test data")
    print(X_test.shape)
    print(y_test.shape)
    
    # Data normalization (0,1)
    X_test = preprocessing.normalize(X_test, norm='l2')
    
    # Models training
    
    # Run model
    clf_lda = load('Inference_lda.joblib')
    print("LDA score and classification:")
    print(clf_lda.score(X_test, y_test))
    print(clf_lda.predict(X_test))
        
    # Run model
    clf_nn = load('Inference_NN.joblib')
    print("NN score and classification:")
    print(clf_nn.score(X_test, y_test))
    print(clf_nn.predict(X_test))
    
if __name__ == '__main__':
    inference()