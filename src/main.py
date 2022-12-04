import data_collections
import data_preprocessing
import modelling
from sklearn.metrics import classification_report

import pandas as pd

from utils import load_config, load_pickle

def train():

    data_collections.main()

    data_preprocessing.main()

    modelling.main()

def evaluate_model_testing(model, fitur_test, y_test):
    #test
    predY = model.predict(fitur_test)
    probs = model.predict_proba(fitur_test)
    probs = probs[:, 1]
    #classification report
    print("Testing Report")
    print(classification_report(y_test, predY))

if __name__ == "__main__":

    config = load_config()

    train()

    model = load_pickle(config["production_model_path"])

    x_test = load_pickle(config["prep_test_path"]["x_test"])
    y_test = load_pickle(config["prep_test_path"]["y_test"])

    evaluate_model_testing(model,x_test,y_test)