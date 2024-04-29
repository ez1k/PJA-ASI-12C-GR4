"""
This is a boilerplate pipeline 'modeling'
generated using Kedro 0.19.2
"""
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

def split_data (data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] :
    X = data.drop('Price', axis=1)
    y = data['Price']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)
    
    return X_train, X_test, X_val, y_train, y_test, y_val




