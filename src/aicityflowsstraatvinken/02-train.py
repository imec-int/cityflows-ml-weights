import warnings



warnings.filterwarnings("ignore")
import os
import pandas as pd
pd.options.mode.chained_assignment = None

from utils import most_recent_dataset
from utils import get_traintest_data
from utils import prepare_model
from utils import report_model
from utils import accuracy_report

from sklearn.linear_model import ElasticNetCV, LarsCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor, XGBRFRegressor
from catboost import CatBoostRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ARDRegression
import yaml
import numpy as np
import pickle
import lightgbm
import time
from datetime import date, datetime

ITERATION = "2022_3"
TRAIN_CONFIG_PATH = f"src/aicityflowsstraatvinken/train_config_{ITERATION}.yaml"
INFER_CONFIG_PATH = TRAIN_CONFIG_PATH.replace("train", "infer")
TRAINING_DAY=date.today().isoformat().replace("-", "")
#TRAINING_DAY="20220304"
RANDOM_STATE=42

def train_models():
    print(f"Training models on {datetime.now().isoformat(timespec='seconds')}")

    scoring={"r2": "r2", "mape": "neg_mean_absolute_percentage_error", "mae": "neg_mean_absolute_error", }
    refit="r2"

    config = yaml.load(stream=open(TRAIN_CONFIG_PATH, 'r'), Loader=yaml.FullLoader)
    print(f" using config of iteration '{config['iteration']}' created on {config['date_created']}")

    Y_s = config["columns"]["Y_s"]
    one_hot_cols = config["columns"]["one_hot_cols"]
    num_pred_remain_cols = config["columns"]["num_pred_remain_cols"]
    num_pred_segm_cols = config["columns"]["num_pred_segm_cols"]
    num_pred_minmax_cols = config["columns"]["num_pred_minmax_cols"]

    if "dataset_path" not in config:
        DATASET_PATH = most_recent_dataset()
    else:
        DATASET_PATH = config["dataset_path"]
    print(f" using dataset {DATASET_PATH}")
    dataset = pickle.load(open(DATASET_PATH, 'rb'))
    X_train, X_test, y_train, y_test = get_traintest_data(dataset, config)
    #print("X_train columns", X_train.columns)

    one_hot_categories = list(dataset[one_hot_cols].astype(str).apply(lambda x: list(set(x)), axis=0).values)
    #print("one hot categories", one_hot_categories)
    prefix = np.hstack([[one_hot_cols[ix]] * len(cat_list) for ix, cat_list in enumerate(one_hot_categories)]) 
    suffix =  np.hstack([cat_list for cat_list in one_hot_categories]).astype(str)
    suffix = [s.replace(".0", "") for s in suffix]
    #print(suffix)
    one_hot_col_names = [f"{prefix}_{suffix}" for prefix, suffix in zip(prefix, suffix)]
    config["columns"]["one_hot_col_names"] = one_hot_col_names
    config["columns"]["one_hot_categories"] = one_hot_categories
    num_pred_minmax = num_pred_minmax_cols.copy()
    num_pred_minmax.extend(num_pred_segm_cols)

    cols = one_hot_cols.copy()
    cols.extend(num_pred_minmax)
    cols.extend(num_pred_remain_cols)

    model_params = [
        (ElasticNetCV(random_state=RANDOM_STATE),
            {
                "est__cv": [10]
            }
        ),
        # (TabNetRegressor(seed=RANDOM_STATE, verbose=0, patience=15),
        #     {
        #         'est__n_a': [8], # max 64
        #         'est__n_d': [8], # preferably same as n_a
        #         'est__n_steps': [2, 5],
        #         'est__gamma': [1.3], #1, 1.3, 1.7]
        #         'est__n_independent': [2, 4],
        #         'est__n_shared': [2, 4]
        #     }
        # ),
        (CatBoostRegressor(loss_function="RMSE", random_state=RANDOM_STATE, silent=True),
            {
                'est__learning_rate': [0.01, 0.05, 0.1],
                'est__depth': [6, 8, 10],
                #'est__l2_leaf_reg': [7, 9],
                'est__iterations': [30, 50, 100]
                #'est__learning_rate': [0.03, 0.1],
                #'est__depth': [3, 6, 8],
                #'est__l2_leaf_reg': [7, 9]
            }
        ),
        (lightgbm.LGBMRegressor(random_state=RANDOM_STATE, silent=True), 
            {
                'est__num_leaves': [7, 14, 21, 28, 31, 50],
                'est__learning_rate': [0.1, 0.03, 0.003],
                'est__max_depth': [-1, 3, 5],
                'est__n_estimators': [50, 100, 200, 500],
            }
        ),
        # (SVR(gamma="scale", shrinking=True),
        #     {
        #         'est__kernel': ['rbf', 'poly', 'linear'],
        #         'est__degree': [2, 3, 5], # only relevant for polynomial kernel
        #         'est__C': [0.5, 1.0, 2.0, 5.0],
        #         'est__epsilon': [0.01, 0.1, 0.5],
        #     }
        # ),
        # (ARDRegression(verbose=False), 
        #     {
        #         'est__n_iter': [100, 200, 300],
        #         'est__tol': [1e-5, 1e-3],
        #         'est__alpha_1': [1e-6],
        #         'est__alpha_2': [1e-6],
        #         'est__lambda_1': [1e-4, 1e-6],
        #         'est__lambda_2': [1e-4, 1e-6],
        #         'est__threshold_lambda': [1e4, 1e6],
        #     }
        # ),
        (LarsCV(),
            {
                "est__verbose": [False]
            }
        ),
        # (MLPRegressor(random_state=RANDOM_STATE, verbose=False),
        #     {
        #         "est__hidden_layer_sizes": [(100,), (100, 50, 25,), (50, 15, )],
        #         "est__solver": ["lbfgs", "adam"],
        #         "est__alpha": [1e-4, 1e-3]
        #     }
        # )
    ]

    begin = time.time()
    

    best_models={}

    for y in Y_s:
        start_y=time.time()
        path_prefix = f"data/model/{TRAINING_DAY}-iteration{config['iteration']}-{y}/"
        os.makedirs(path_prefix, exist_ok=True)
        #TODO: keep track of best model based on "refit" metric
        print(f"training models for target {y}")
        best_acc=0
        for model, params in model_params:
            
            model_name = type(model).__name__
            print(f" - training model {model_name}")
            model_serialization_path = f"{path_prefix}{TRAINING_DAY}-{model_name}-model-{y}.pkl"
            if os.path.exists(model_serialization_path):
                gs = pickle.load(open(model_serialization_path, "rb"))
                acc=accuracy_report(gs, X_test[cols], y_test[y], summary=True)
                if acc>best_acc:
                    best_estimator = model_name
                    best_estimator_path = model_serialization_path
                    best_acc=acc
                print("model already trained! file:", model_serialization_path, 'accuracy', acc)
                
                continue
            start=time.time()
            # prepare gridsearch
            gs = prepare_model(model, config, params, scoring, refit)
            y_tr = y_train[y]
            if model_name == "TabNetRegressor":
                y_tr=y_tr.to_numpy().reshape(-1, 1)
            gs.fit(X_train[cols], y_tr)
            _model = gs.best_estimator_
            config_output_path = f"{path_prefix}{TRAINING_DAY}-{model_name}-config-{y}.yaml"
            report_path = f"{path_prefix}{TRAINING_DAY}-{model_name}-report-{y}.txt"
            pickle.dump(_model, open(model_serialization_path, "wb"))
            model_data = {
                "id": f"{datetime.now().isoformat()}-{model_name}-{y}", 
                "name": model_name,
                "training_date": datetime.now().isoformat(timespec="seconds"),
                "params": gs.best_params_,
                "traffic_modality": y,
                "report_path": report_path
            }
            config["model"] = model_data
            config["config_path"] = config_output_path
            yaml.dump(config, stream=open(config_output_path, 'w'), default_flow_style=False, sort_keys=False)
            #write report
            with open(report_path, "w") as report:
                report.writelines(report_model(gs, X_train[cols], y_train[y], X_test[cols], y_test[y], config))
            # including explain_model() # TODO, specific functions for some models
            acc=accuracy_report(gs, X_test[cols], y_test[y], summary=True)
            if acc>best_acc:
                best_estimator = model_name
                best_estimator_path = model_serialization_path
                best_acc=acc
            print(f"     - training model {model_name} got score {acc:.4f} and took {time.time()-start:.3f}s")


        print(f" in total took {time.time()-start_y:.3f}s best estimator {best_estimator} with accuracy {best_acc:.4f}")
        best_models[y] = {
            "model": best_estimator,
            "model_path": best_estimator_path
        }

    #TODO: automatically create inference config infer_config_{config['iteration']}.yaml
    # with best performing models
    
    infer_config = {
        "iteration": config["iteration"],
        "date_created": date.today().isoformat().replace("-", ""),
        "infer_models": best_models,
        "columns": config["columns"]
    }
    yaml.dump(infer_config, stream=open(INFER_CONFIG_PATH, 'w'), default_flow_style=False, sort_keys=False)
    print(f"inference config written to {INFER_CONFIG_PATH}")

    print(f"complete training took {time.time()-begin:.3f}s")

if __name__ == "__main__":
    train_models()