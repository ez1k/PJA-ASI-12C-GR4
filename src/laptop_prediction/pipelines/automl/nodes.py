"""
This is a boilerplate pipeline 'automl'
generated using Kedro 0.19.2
"""
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor

def train_model_champion(train_data: pd.DataFrame):
    print(train_data)
    predictor = TabularPredictor(label='Price').fit(train_data)
    return predictor

def train_model_challenger(train_data: pd.DataFrame):
    print(train_data)
    predictor = TabularPredictor(label='Price').fit(train_data, hyperparameters={
        'GBM': {'num_boost_round': 100},
        'NN_TORCH': {'epochs': 10},
        'CAT': {'iterations': 200}
    })
    return predictor

def evaluate_models(model_champion, model_challenger, test_data: pd.DataFrame):
    score_champion = model_champion.evaluate(test_data)['root_mean_squared_error'] * -1
    score_challenger = model_challenger.evaluate(test_data)['root_mean_squared_error'] * -1
    if score_challenger < score_champion:  # Assuming lower RMSE is better
        best_model_ml = model_champion
        message = f"Champion model is better with RMSE {score_champion:.4f} compared to Challenger model with RMSE {score_challenger:.4f}."
    else:
        best_model_ml = model_challenger
        message = f"Challenger model is better with RMSE {score_challenger:.4f} compared to Champion model with RMSE {score_champion:.4f}."

    print(message)
    return best_model_ml