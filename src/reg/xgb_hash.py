import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.feature_extraction import FeatureHasher
from joblib import dump
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# Load the dataset
df = pd.read_csv('../input/hp_combined_1.csv')  # Make sure to use the correct path

# Remove outliers
df = df[df['PCF'] <= 3500]

# Define features
columns = ['Commercial Name', 'Processor Cores', 'Memory', 'SSD', 'SSD Brand', 'Power', 'Graphics', 'Audio', 'PCF']
df_selected = df[columns]

X = df_selected.drop('PCF', axis=1)
y = df_selected['PCF']

# Define categorical and numeric features
numeric_features = ['Processor Cores', 'Memory', 'SSD', 'Power']
categorical_features = ['Commercial Name', 'SSD Brand', 'Graphics', 'Audio']

# Function to apply hashing on a dataframe
def hash_features(df, n_features=10):
    hasher = FeatureHasher(n_features=n_features, input_type='string')
    df_as_dicts = df.astype(str).to_dict(orient='records')
    hashed_features = hasher.transform(df_as_dicts)
    return hashed_features.toarray()

# Hashing transformer
hashing_transformer = FunctionTransformer(hash_features, kw_args={'n_features': 10}, validate=False)

# ColumnTransformer for numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        # Note: Hashing is applied separately
    ])

# Preprocess and split the dataset
X_numeric_transformed = preprocessor.fit_transform(X[numeric_features])
X_categorical_hashed = hashing_transformer.transform(X[categorical_features])

# Combine numeric and hashed categorical features
X_combined = np.hstack((X_numeric_transformed, X_categorical_hashed))

X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.25, random_state=42)

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
