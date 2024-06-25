"""
This is a boilerplate pipeline 'automl'
generated using Kedro 0.19.2
"""
import pandas as pd
from autogluon.tabular import TabularPredictor

def train_model_challenger(train_data: pd.DataFrame):
    print(train_data)
    predictor = TabularPredictor(label='Price').fit(train_data, 
    hyperparameters={
        'GBM': {'num_boost_round': 100},
        'NN_TORCH': {'epochs': 10},
        'CAT': {'iterations': 200}
    }
    )
    return predictor

def evaluate_model(model, test_data: pd.DataFrame):
    evaluation_results = model.evaluate(test_data)
    print(evaluation_results)

    val_mae = evaluation_results.get('mean_absolute_error') * -1
    val_mse = evaluation_results.get('mean_squared_error') * -1
    val_rmse = evaluation_results.get('root_mean_squared_error') * -1
    val_r2 = evaluation_results.get('r2')
 
    metrics = {
        'val_mae': val_mae,
        'val_mse': val_mse,
        'val_rmse': val_rmse,
        'val_r2': val_r2
    }
    
    return metrics