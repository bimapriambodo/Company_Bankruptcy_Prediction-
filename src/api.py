from fastapi import FastAPI
from pydantic import BaseModel
import copy
import uvicorn
import pandas as pd
import numpy as np

import src.utils as utils

# read config
config = utils.load_config()

# load model
std_scaler = utils.load_pickle(config["scaler_path"])
prod_model = utils.load_pickle(config["production_model_path"])

# define input data
class InputData(BaseModel):
    Attr21: float
    Attr13: float
    Attr27: float
    Attr34: float
    Attr24: float
    Attr35: float
    Attr51: float
    Attr6: float
    Attr56: float
    Attr49: float
    Attr40: float
    Attr55: float
    Attr50: float
    Attr58: float
    Attr43: float

def std_scaler_transform(features: pd.DataFrame, scaler: object) -> pd.DataFrame:
    '''
    this function transform features using standar scaler machine
    '''

    col_names = scaler.feature_names_in_

    feat = copy.deepcopy(features)

    scaled = scaler.transform(feat)

    scaled_df = pd.DataFrame(scaled, columns=col_names)

    return scaled_df

app = FastAPI()

# landing page
@app.get("/")
def home():
    return "Hello, FastAPI up!"

# prediction page
@app.post("/predict/")
def predict(data: InputData):
    # Convert data api to dataframe
    data = pd.DataFrame(data).set_index(0).T.reset_index(drop=True)

    # # convert to float
    data = data.astype("float64")

    # standard scaling
    data = std_scaler_transform(data, std_scaler)

    # Predict data
    y_pred = str(prod_model.predict(data))[1]

    return {"prediction" : y_pred, "error_msg": ""}

if __name__ == "__main__":
    uvicorn.run("api:app", host = "0.0.0.0", port = 8080)