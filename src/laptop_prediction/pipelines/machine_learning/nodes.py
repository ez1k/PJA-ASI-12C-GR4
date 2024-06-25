import os
import pandas as pd
import numpy as np

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
import wandb
import joblib


def run_model(X_train: pd.DataFrame, y_train: pd.DataFrame, forest_n: int) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=forest_n, random_state=0)
    model.fit(X_train, y_train.values.ravel())
    return model

def optimize_model(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.DataFrame, cv: any, verbose: int, n_jobs: int) -> RandomForestRegressor:
    params = {
        'n_estimators': [int(x) for x in np.linspace(start=10, stop=200, num=10)],
        'max_depth': [int(x) for x in np.linspace(2, 30, num=5)] + [None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    random_search = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=100, cv=cv, verbose=verbose, random_state=0, n_jobs=n_jobs)
    random_search.fit(X_test, y_test.values.ravel())
    best_model = random_search.best_estimator_
    return best_model

def compare_with_old_model(new_model: RandomForestRegressor, X_val: pd.DataFrame, y_val: pd.DataFrame, retraining: bool):
    model_path = 'data/04_model/best_model.pickle'

    if(not retraining):
        return new_model
    if(not os.path.exists(model_path)):
        print("Old model not available, skipping.")
        return new_model
    
    y_pred_test_new_moddel = new_model.predict(X_val)
    score_new_model = mean_squared_error(y_val, y_pred_test_new_moddel)

    old_model = joblib.load(model_path)

    y_pred_test_old_model = old_model.predict(X_val)
    score_old_model = mean_squared_error(y_val, y_pred_test_old_model)

    result_model = new_model

    if score_new_model < score_old_model:
        message = "New model is better."
    else:
        result_model = old_model
        message = "Old model is better."
    
    print(message)

    return result_model

def evaluate_model(best_model: RandomForestRegressor, X_val: pd.DataFrame, y_val: pd.DataFrame) -> dict:
    metrics = {}

    y_pred_test = best_model.predict(X_val)
    metrics['val_mae'] = mean_absolute_error(y_val, y_pred_test)
    metrics['val_mse'] = mean_squared_error(y_val, y_pred_test)
    metrics['val_rmse'] = math.sqrt(metrics['val_mse'])
    metrics['val_r2'] = r2_score(y_val, y_pred_test)   

    wandb.init(project='PJA-ASI-12C-GR4')
    wandb.log({
        "Val MAE": metrics['val_mae'],
        "Val MSE": metrics['val_mse'],
        "Val RMSE": metrics['val_rmse'],
        "Val R2": metrics['val_r2'],
    })

    return metrics