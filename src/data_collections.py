import pandas as pd
import copy
from utils import load_config, dump_pickle
from scipy.io import arff
from sklearn.model_selection import train_test_split

config = load_config()

def read_raw_df(raw_data_path: str):
    """
    Buat ngambil data raw cui
    """
    df = arff.loadarff(raw_data_path)
    df = pd.DataFrame(df[0])
    
    return df

def cleansing(dataframe: pd.DataFrame)->pd.DataFrame:
    """
    ini fungsi buat bersihin data kotor yang kagak jelas
    """
    temp = copy.deepcopy(dataframe)
    # temp.dropna(inplace=True)
    temp['class'] = temp['class'].replace("b",'')
    temp['class'] = temp['class'].astype(int)
    return temp

def split_data(dataframe: pd.DataFrame) -> tuple:
    """
    kalo fungsi ini buat split data test dan data train cui
    """
    df = copy.deepcopy(dataframe)

    X = df.drop(columns=config["label"])
    y = df[config["label"]]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    return x_train, x_test, y_train, y_test

def imputation_selections(x_train: pd.DataFrame, x_test: pd.DataFrame)->pd.DataFrame:
    """
    ini fungsi buat handling missing valus dan milih variable yang pengen di training
    """
    x_train = copy.deepcopy(x_train)
    x_test = copy.deepcopy(x_test)
    
    #select variables using modelling
    x_train = x_train[config["predictors"]]
    x_test = x_test[config["predictors"]]
    
    #handling missing values
    x_train.fillna(-9999999999, inplace=True)
    x_test.fillna(-9999999999, inplace=True)
    
    return x_train, x_test

def int_to_float(dataframe: pd.DataFrame)->pd.DataFrame:
    """
    ini fungsi buat mastiin kalau datanya harus punya tipe data float
    """
    df = copy.deepcopy(dataframe)

    df[config["float_columns"]] = df[config["float_columns"]].astype("float64")

    return df

def main():
    """
    Jalankan pipeline data collections
    """
    config = load_config()

    df = read_raw_df(config["raw_dataset_path"])
    dump_pickle(df, config["raw_df_path"])
    df = cleansing(df)
    x_train, x_test, y_train, y_test = split_data(df)
    x_train, x_test = imputation_selections(x_train, x_test)
    x_train = int_to_float(x_train)
    x_test = int_to_float(x_test)
    dump_pickle(x_train, config["train_set_path"]["x_train"])
    dump_pickle(y_train, config["train_set_path"]["y_train"])
    dump_pickle(x_test, config["test_set_path"]["x_test"])
    dump_pickle(y_test, config["test_set_path"]["y_test"])

if __name__ == "__main__":
    config = load_config()
    main()
    print("data collections success!")