"""
This is a boilerplate pipeline 'ml'
generated using Kedro 0.19.2
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
import wandb

def transform_data(laptop_data: pd.DataFrame) -> pd.DataFrame:
    laptop_data.drop('Unnamed: 0', axis=1, inplace=True)
    laptop_data['Ram'] = laptop_data['Ram'].str.replace('GB', '').astype(int)
    laptop_data['Weight'] = laptop_data['Weight'].str.replace('kg', '').astype(float)

    is_fullhd = laptop_data['ScreenResolution'].str.contains('Full HD')
    laptop_data['IsFullHD'] = is_fullhd.astype(int)

    is_ips = laptop_data['ScreenResolution'].str.contains('IPS')
    laptop_data['IsIPS'] = is_ips.astype(int)

    is_touchscreen = laptop_data['ScreenResolution'].str.contains('Touchscreen')
    laptop_data['IsTouchscreen'] = is_touchscreen.astype(int)

    is_retina = laptop_data['ScreenResolution'].str.contains('Retina')
    laptop_data['IsRetina'] = is_retina.astype(int)

    resolution = laptop_data['ScreenResolution'].str.extract(r'(\d+)x(\d+)')
    laptop_data['ResolutionWidth'] = resolution[0].astype(int)
    laptop_data['ResolutionHeight'] = resolution[1].astype(int)

    is_hdd = laptop_data['Memory'].str.contains('HDD')
    laptop_data['IsHDD'] = is_hdd.astype(int)

    is_ssd = laptop_data['Memory'].str.contains('SSD')
    laptop_data['IsSSD'] = is_ssd.astype(int)

    is_flashstorage = laptop_data['Memory'].str.contains('Flash Storage')
    laptop_data['IsFlashStorage'] = is_flashstorage.astype(int)

    memory_size_hdd = laptop_data['Memory'].str.extract(r'(\d+)TB HDD')
    memory_size_hdd_gb = laptop_data['Memory'].str.extract(r'(\d+)GB HDD')
    memory_size_ssd = laptop_data['Memory'].str.extract(r'(\d+)GB SSD')
    flash_storage = laptop_data['Memory'].str.extract(r'(\d+)GB Flash Storage')

    laptop_data['FlashStorage'] = flash_storage[0].astype(float)
    laptop_data['MemorySizeHDD_TB'] = memory_size_hdd[0].astype(float)
    laptop_data['MemorySizeHDD_GB'] = memory_size_hdd_gb[0].astype(float)
    laptop_data['MemorySizeSSD'] = memory_size_ssd[0].astype(float)

    cpu_speed = laptop_data['Cpu'].str.extract(r'(\d+\.\d+)GHz')
    cpu_speed = cpu_speed.fillna(laptop_data['Cpu'].str.extract(r'(\d+)GHz'))
    laptop_data['CpuSpeed'] = cpu_speed[0].astype(float)

    cpu_brand = laptop_data['Cpu'].str.extract(r'([A-Za-z]+)')
    laptop_data['CpuBrand'] = cpu_brand[0]

    gpu_brand = laptop_data['Gpu'].str.extract(r'([A-Za-z]+)')
    laptop_data['GpuBrand'] = gpu_brand[0]

    laptop_data['OpSys'] = laptop_data['OpSys'].str.lower()
    laptop_data['OperatingSystem'] = laptop_data['OpSys'].apply(lambda x: 'macos' if 'mac' in x else (
        'windows' if 'windows' in x else ('linux' if 'linux' in x else 'freedos/other')))
    
    laptop_data['Price'] = laptop_data['Price'].astype(int)
 
    laptop_data.drop(['ScreenResolution', 'Memory', 'OpSys'], axis=1, inplace=True)

    return laptop_data

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

    return str(metrics)

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = data.select_dtypes(include=['object']).columns
    label_encoder = LabelEncoder()
    for col in categorical_cols:
        data[col] = label_encoder.fit_transform(data[col])

    data['FlashStorage'] = data['FlashStorage'].fillna(0)
    data['MemorySizeHDD_TB'] = data['MemorySizeHDD_TB'].fillna(0)
    data['MemorySizeHDD_GB'] = data['MemorySizeHDD_GB'].fillna(0)
    data['MemorySizeSSD'] = data['MemorySizeSSD'].fillna(0)

    return data

def split_data (data: pd.DataFrame, train_size: float, val_size: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] :
    X = data.drop('Price', axis=1)
    y = data['Price']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=train_size, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=0)
    
    return X_train, X_test, X_val, y_train, y_test, y_val