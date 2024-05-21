"""
This is a boilerplate pipeline 'automl'
generated using Kedro 0.19.2
"""
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor

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

def split_data(data: pd.DataFrame, test_size: float):
    train_data, test_data = train_test_split(data, test_size=test_size)
    return train_data, test_data

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
    score_champion = abs(model_champion.evaluate(test_data)['root_mean_squared_error'])
    score_challenger = abs(model_challenger.evaluate(test_data)['root_mean_squared_error'])
    if score_challenger < score_champion:  # Assuming lower RMSE is better
        best_model = model_challenger
        message = f"Challenger model is better."
    else:
        best_model = model_champion
        message = f"Champion model is better."

    print(message)
    return best_model