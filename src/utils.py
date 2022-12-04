import yaml
import joblib

config_path = r"config/config.yaml"

def load_config()->dict:

    '''
    this function load the config.yaml file
    '''

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    except FileNotFoundError as fe:
        raise RuntimeError("Parameter File Not Found!")

    return config

def load_pickle(file_path: str):

    '''
    handler function to load pickle file
    '''
    try:
        with open(file_path, "rb") as f:
            pick = joblib.load(f)

    except FileNotFoundError as fe:
        raise RuntimeError("Parameter File Not Found!")

    return pick

def dump_pickle(file, file_path: str):

    '''
    handler function to dump pickle
    '''

    return joblib.dump(file, file_path)


if __name__ == "__main__":

    config_path = r"../config/config.yaml"

    config = load_config(config_path)