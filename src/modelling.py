from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from datetime import datetime
from tqdm import tqdm
import yaml
import joblib
import json
import pandas as pd
import copy
import hashlib

from utils import load_config, load_pickle, dump_pickle

config = load_config()

def time_stamp():
    """
    fungsi waktu, waktu hari ini
    """
    return datetime.now()

def create_log_template():
    """
    fungsi buat bikin template nyimpen hasil training model dalam bentuk json
    """
    logger = {
        "model_name" : [],
        "model_uid" : [],
        "training_time" : [],
        "training_date" : [],
        "performance" : [],
        "f1_score_avg" : [],
        "precision_avg" : [],
        "recall_avg" : [],
        "data_configurations" : [],
    }
    return logger

def training_log_updater(current_log, log_path):
    """
    kalo yang ini buat ngeupdate logger yang telah dibikin
    """
    current_log = current_log.copy()

    try:
        with open(log_path, "r") as file:
            last_log = json.load(file)
        file.close()
    except FileNotFoundError as ffe:
        with open(log_path, "w") as file:
            file.write("[]")
        file.close()
        with open(log_path, "r") as file:
            last_log = json.load(file)
        file.close()
    
    last_log.append(current_log)

    with open(log_path, "w") as file:
        json.dump(last_log, file)
        file.close()

    return last_log

def train_eval_model(list_of_model, prefix_model_name, x_train, y_train, data_configuration_name, x_valid, y_valid, log_path):
    """
    core of the core nih, ini fungsi buat training si model
    """

    list_of_model = copy.deepcopy(list_of_model)
    logger = create_log_template()

    for model in tqdm(list_of_model):    
        model_name = prefix_model_name + "-" + model["model_name"]

        start_time = time_stamp()
        model["model_object"].fit(x_train, y_train)
        finished_time = time_stamp()

        elapsed_time = finished_time - start_time
        elapsed_time = elapsed_time.total_seconds()

        y_pred = model["model_object"].predict(x_valid)
        performance = classification_report(y_valid, y_pred, output_dict = True)

        plain_id = str(start_time) + str(finished_time)
        chiper_id = hashlib.md5(plain_id.encode()).hexdigest()

        model["model_uid"] = chiper_id

        logger["model_name"].append(model_name)
        logger["model_uid"].append(chiper_id)
        logger["training_time"].append(elapsed_time)
        logger["training_date"].append(str(start_time))
        logger["performance"].append(performance)
        logger["f1_score_avg"].append(performance["macro avg"]["f1-score"])
        logger["precision_avg"].append(performance["macro avg"]["precision"])
        logger["recall_avg"].append(performance["macro avg"]["recall"])
        logger["data_configurations"].append(data_configuration_name)

    training_log = training_log_updater(logger, log_path)

    return training_log, list_of_model

def training_log_to_df(training_log):
    """
    buat convert hasil training model ke dataframe pandas
    """
    training_res = pd.DataFrame()

    for log in tqdm(training_log):
        training_res = pd.concat([training_res, pd.DataFrame(log)])
    
    training_res.sort_values(["f1_score_avg", "training_time"], ascending = [False, True], inplace = True)
    training_res.reset_index(inplace = True, drop = True)
    
    return training_res

def get_best_model(training_log_df, list_of_model):
    """
    Nah kalau fungsi ini buat milih model mana yang paling bagus, dan hasilnya buat production di api nantinya
    """
    model_object = None

    best_model_info = training_log_df.sort_values(["f1_score_avg", "training_time"], ascending = [False, True]).iloc[0]
    i=0
    for configuration_data in list_of_model:
        for model_data in list_of_model[configuration_data]:
            if model_data["model_uid"] == best_model_info["model_uid"]:
                model_object = model_data["model_object"]
                break

    if model_object == None:
        model_object = load_pickle(config["production_model_path"])
        print("The best model not found in your list of model.")
    
    return model_object

def main():
    """
    jelas dong, buat ngejalanin fungsi2 yang ribet di atas
    """
    config = load_config()
    # load preprocessed pickle of dataset
    x_rus = load_pickle(config["prep_rus_path"]["x_rus"])
    y_rus = load_pickle(config["prep_rus_path"]["y_rus"])

    x_sm = load_pickle(config["prep_sm_path"]["x_sm"])
    y_sm = load_pickle(config["prep_sm_path"]["y_sm"])

    x_test = load_pickle(config["prep_test_path"]["x_test"])
    y_test = load_pickle(config["prep_test_path"]["y_test"])
    
    # initiate models
    lgr_baseline = LogisticRegression()
    dct_baseline = DecisionTreeClassifier()
    rfc_baseline = RandomForestClassifier()
    knn_baseline = KNeighborsClassifier()
    xgb_baseline = XGBClassifier(seed=42)
    xgb_tuning = XGBClassifier(max_depth=9,
                               min_child_weight=1,
                               gamma=0.0,
                               reg_alpha=1e-05,
                               seed=42)
    
        # list of models
    list_of_model = {
    "undersampling" : [
        { "model_name": lgr_baseline.__class__.__name__, "model_object": lgr_baseline, "model_uid": ""},
        { "model_name": dct_baseline.__class__.__name__, "model_object": dct_baseline, "model_uid": ""},
        { "model_name": rfc_baseline.__class__.__name__, "model_object": rfc_baseline, "model_uid": ""},
        { "model_name": knn_baseline.__class__.__name__, "model_object": knn_baseline, "model_uid": ""},
        { "model_name": xgb_baseline.__class__.__name__, "model_object": xgb_baseline, "model_uid": ""},
        { "model_name": xgb_tuning.__class__.__name__+"_"+"tuning", "model_object": xgb_tuning, "model_uid": ""}
        ],
    "smote" : [
        { "model_name": lgr_baseline.__class__.__name__, "model_object": lgr_baseline, "model_uid": ""},
        { "model_name": dct_baseline.__class__.__name__, "model_object": dct_baseline, "model_uid": ""},
        { "model_name": rfc_baseline.__class__.__name__, "model_object": rfc_baseline, "model_uid": ""},
        { "model_name": knn_baseline.__class__.__name__, "model_object": knn_baseline, "model_uid": ""},
        { "model_name": xgb_baseline.__class__.__name__, "model_object": xgb_baseline, "model_uid": ""},
        { "model_name": xgb_tuning.__class__.__name__+"_"+"tuning", "model_object": xgb_tuning, "model_uid": ""}
        ],
    }
        # training of rus
    training_log, list_of_model_rus = train_eval_model(
    list_of_model["undersampling"],
    "baseline_model",
    x_rus,
    y_rus,
    "undersampling",
    x_test,
    y_test,
    config["training_log_path"]
    )
    
    list_of_model["undersampling"] = copy.deepcopy(list_of_model_rus)
    
       # training of smote
    training_log, list_of_model_sm = train_eval_model(
    list_of_model["smote"],
    "baseline_model",
    x_sm,
    y_sm,
    "smote",
    x_test,
    y_test,
    config["training_log_path"]
    )
    
    list_of_model["smote"] = copy.deepcopy(list_of_model_sm)
    
    # log to df
    training_res = training_log_to_df(training_log)
    
    # get best model
    model = get_best_model(training_res, list_of_model)
    
    # save best model
    dump_pickle(model, config["production_model_path"])

if __name__ == "__main__":
    config = load_config()
    main()
    print("modelling success!!!")