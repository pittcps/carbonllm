import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from joblib import dump
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# Load the dataset
df = pd.read_csv('../input/hp_combined_hdd.csv')

# Remove outliers
df = df[df['PCF'] <= 3500]

# Define features
columns = ['Processor Cores', 'Memory', 'SSD', 'Power', 'HDD', 'PCF']
df_selected = df[columns]

X = df_selected.drop('PCF', axis=1)
y = df_selected['PCF']

# Define numeric features
numeric_features = ['Processor Cores', 'Memory', 'SSD', 'Power', 'HDD']

# ColumnTransformer for numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), # drop these data points; give 0s
    ('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# Preprocess the dataset
X_transformed = preprocessor.fit_transform(X[numeric_features])

X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.25, random_state=42)

# Train the model
model = XGBRegressor(objective='reg:squarederror')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f'MSE: {mse}, MAE: {mae}, MAPE: {mape}')

# Plot test vs. predicted data points
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted PCF')
plt.savefig('output/xgb_m_carbon_plot.png')

# Save the model
dump(model, 'output/reg_model.joblib')
dump(preprocessor, 'output/preprocessor.joblib')
