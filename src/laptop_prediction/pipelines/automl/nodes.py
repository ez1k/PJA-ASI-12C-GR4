"""
This is a boilerplate pipeline 'automl'
generated using Kedro 0.19.2
"""
import json
import os
import pickle
import pandas as pd
from autogluon.tabular import TabularPredictor

def train_model_champion(train_data: pd.DataFrame):
    print(train_data)
    predictor = TabularPredictor(label='Price').fit(train_data, 
    # hyperparameters={
    #     'GBM': {'num_boost_round': 100},
    #     'NN_TORCH': {'epochs': 10},
    #     'CAT': {'iterations': 200}
    # }
    )
    return predictor

def evaluate_model(model, test_data: pd.DataFrame):
    score = model.evaluate(test_data)['root_mean_squared_error']
    return score

def compare_with_challenger(score_challenger, model_challenger):
    model_path = "data/04_model/best_model_aml.pkl"
    score_path = "data/05_model_output/best_model_aml_score.json"
    
    os.makedirs(os.path.dirname(score_path), exist_ok=True)
    
    if os.path.exists(model_path) and os.path.exists(score_path):
        with open(score_path, "r") as f:
            best_model_aml_score = json.load(f)
        if score_challenger < best_model_aml_score:
            best_model_aml_score = score_challenger
            best_model_aml = model_challenger
            with open(model_path, "wb") as f:
                pickle.dump(best_model_aml, f)
            with open(score_path, "w") as f:
                json.dump(best_model_aml_score, f)
            message = "New challenger model is better. Updated the champion model."
        else:
            with open(model_path, "rb") as f:
                best_model_aml = pickle.load(f)
            message = "Champion model is still better. No update made."
    else:
        best_model_aml_score = score_challenger
        best_model_aml = model_challenger
        with open(model_path, "wb") as f:
            pickle.dump(best_model_aml, f)
        with open(score_path, "w") as f:
            json.dump(best_model_aml_score, f)
        message = "No existing champion model. Set the challenger as the champion model."
    
    print(message)
    return best_model_aml, best_model_aml_score