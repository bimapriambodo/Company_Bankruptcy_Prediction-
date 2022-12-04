from sklearn.preprocessing import StandardScaler
import pandas as pd
import copy
from utils import load_config, load_pickle, dump_pickle
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

config = load_config()

def std_scaler_fit(x_train: pd.DataFrame):
    """
    ini fungsi buat fitting scaler
    """
    std_scaler = StandardScaler()

    std_scaler.fit(x_train)

    return std_scaler

def std_scaler_transform(features: pd.DataFrame, scaler: object) -> pd.DataFrame:

    '''
    this function transform features using standar scaler machine
    '''
    
    col_names = scaler.feature_names_in_

    feat = copy.deepcopy(features)

    scaled = scaler.transform(feat)

    scaled_df = pd.DataFrame(scaled, columns=col_names)

    return scaled_df

def rus_balancer(x_train, y_train):
    """
    ini fungsi buat handling imbalance data, tapi pakai undersampling ya!
    """
    rus = RandomUnderSampler(random_state=42)

    x_rus, y_rus = rus.fit_resample(x_train, y_train)

    return x_rus, y_rus

def sm_balancer(x_train, y_train):
    """
    sama aja si kayak fungsi di atas, tapo ini oversampling pakai smote
    """
    sm = SMOTE(random_state=42)

    x_sm, y_sm = sm.fit_resample(x_train, y_train)

    return x_sm, y_sm

def main():
    """
    kalao yang ini jelas jalanin pipeline fungsi2 di atas!
    """
    config = load_config()

    x_train = load_pickle(config["train_set_path"]["x_train"])
    x_test = load_pickle(config["test_set_path"]["x_test"])

    y_train = load_pickle(config["train_set_path"]["y_train"])
    y_test = load_pickle(config["test_set_path"]["y_test"])
    
    # standardizing
    scaler = std_scaler_fit(x_train)
    dump_pickle(scaler, config["scaler_path"])
    x_train_scaled = std_scaler_transform(x_train, scaler)
    x_test_scaled = std_scaler_transform(x_test, scaler)
    # class balancer - rus
    x_rus, y_rus = rus_balancer(x_train_scaled, y_train)
    # class balancer - smote
    x_sm, y_sm = sm_balancer(x_train_scaled, y_train)
    # dump everything
    dump_pickle(x_rus, config["prep_rus_path"]["x_rus"])
    dump_pickle(y_rus, config["prep_rus_path"]["y_rus"])

    dump_pickle(x_sm, config["prep_sm_path"]["x_sm"])
    dump_pickle(y_sm, config["prep_sm_path"]["y_sm"])

    dump_pickle(x_test_scaled, config["prep_test_path"]["x_test"])

if __name__ == "__main__":
    config = load_config()
    main()
    print("data preprocessing success!")