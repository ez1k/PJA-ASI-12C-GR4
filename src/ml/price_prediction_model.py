import numpy as np 
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


laptop_data = pd.read_csv('docs/laptop_data_output_file.csv')

categorical_cols = laptop_data.select_dtypes(include=['object']).columns

label_encoder = LabelEncoder()
for col in categorical_cols:
    laptop_data[col] = label_encoder.fit_transform(laptop_data[col])

laptop_data['FlashStorage'] = laptop_data['FlashStorage'].fillna(0)
laptop_data['MemorySizeHDD_TB'] = laptop_data['MemorySizeHDD_TB'].fillna(0)
laptop_data['MemorySizeHDD_GB'] = laptop_data['MemorySizeHDD_GB'].fillna(0)
laptop_data['MemorySizeSSD'] = laptop_data['MemorySizeSSD'].fillna(0)

X = laptop_data.drop('Price', axis=1)
y = laptop_data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

params = {
    'n_estimators': [int(x) for x in np.linspace(start=10, stop=200, num=10)],
    'max_depth': [int(x) for x in np.linspace(2, 30, num=5)] + [None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

random_search = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=100, cv=3, verbose=2, random_state=0, n_jobs=-1)
random_search.fit(X_train, y_train)

best_params = random_search.best_params_
print(f"Best parameters: {best_params}")

best_model = RandomForestRegressor(**best_params)
best_model.fit(X_train, y_train)

best_predictions = best_model.predict(X_test)
best_mae = mean_absolute_error(y_test, best_predictions)
print(f"Mean Absolute Error with best model: {best_mae}")
print("Best model MAE vs original model MAE: ", best_mae, mae)