FROM jupyter/scipy-notebook

RUN mkdir model data src
ENV MODEL_DIR=/home/model
ENV MODEL_FILE_GB=gb_clf.joblib
ENV MODEL_FILE_RF=rf_clf.joblib

RUN pip install joblib

COPY ./data/titanic_train_preprocessed.csv ./data/titanic_train_preprocessed.csv
COPY ./data/titanic_test_preprocessed.csv ./data/titanic_test_preprocessed.csv

COPY ./src/train.py ./src/train.py
COPY ./src/inference.py ./src/inference.py

RUN python3 train.py
