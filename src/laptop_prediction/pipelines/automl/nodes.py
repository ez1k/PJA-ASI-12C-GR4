"""
This is a boilerplate pipeline 'automl'
generated using Kedro 0.19.2
"""
import os
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
import pickle
import json
from autogluon.tabular import TabularPredictor

def train_model_challenger(train_data: pd.DataFrame):
    predictor = TabularPredictor(label='Price').fit(train_data, 
    hyperparameters={
        'GBM': {'num_boost_round': 100},
        'NN_TORCH': {'epochs': 10},
        'CAT': {'iterations': 200}
    }
    )
    return predictor

def evaluate_model(model, test_data: pd.DataFrame, retraining: bool, X_test: pd.DataFrame, y_test: pd.DataFrame):
    if (retraining):
        best_model_path = 'data/04_model/best_model.pickle'
        best_score_path = "data/05_model_output/score_best_model.json"
        if os.path.exists(best_model_path) and os.path.exists(best_score_path):
            with open(best_model_path, 'rb') as file:
                best_model = pickle.load(file)
            with open(best_score_path, "r") as f:
                best_model_score = json.load(f)
            champion_model = best_model_score['model']
            print('Retraining')
            if (champion_model == 'ml'):
                y_pred_test = best_model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred_test)
                mse = mean_squared_error(y_test, y_pred_test)
                rmse = math.sqrt(mse)
                r2 = r2_score(y_test, y_pred_test)

                print(champion_model)
                print(mae)
                print(mse)
                print(rmse)
                print(r2)

                retraining_results = {
                    "model": champion_model,
                    "val_mae": mae,
                    "val_mse": mse,
                    "val_rmse": rmse,
                    "val_r2": r2
                }
            else:
                evaluation_results = best_model.evaluate(test_data)
                mae = evaluation_results.get('mean_absolute_error') * -1
                mse = evaluation_results.get('mean_squared_error') * -1
                rmse = evaluation_results.get('root_mean_squared_error') * -1
                r2 = evaluation_results.get('r2')
            
                print(champion_model)
                print(mae)
                print(mse)
                print(rmse)
                print(r2)
            
                retraining_results = {
                    "model": champion_model,
                    "val_mae": mae,
                    "val_mse": mse,
                    "val_rmse": rmse,
                    "val_r2": r2
                }
            with open(best_score_path, "w") as f:
                json.dump(retraining_results, f)   

    evaluation_results = model.evaluate(test_data)

    model = 'automl'
    val_mae = evaluation_results.get('mean_absolute_error') * -1
    val_mse = evaluation_results.get('mean_squared_error') * -1
    val_rmse = evaluation_results.get('root_mean_squared_error') * -1
    val_r2 = evaluation_results.get('r2')
 
    metrics = {
        'model': model,
        'val_mae': val_mae,
        'val_mse': val_mse,
        'val_rmse': val_rmse,
        'val_r2': val_r2
    }
    
    return metrics