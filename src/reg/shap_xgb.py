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
import shap

# Load the dataset
df = pd.read_csv('../input/hp_combined_hdd.csv')

# Remove outliers where 'PCF' <= 3500
df = df[df['PCF'] <= 3500]

# Define features
columns = ['Commercial Name', 'Processor Cores', 'Memory', 'SSD', 'SSD Brand', 'Power', 'HDD', 'Graphics', 'Audio', 'PCF']
df_selected = df[columns]

X = df_selected.drop('PCF', axis=1)
y = df_selected['PCF']

# Define categorical and numeric features
numeric_features = ['Processor Cores', 'Memory', 'SSD', 'Power', 'HDD']
categorical_features = ['Commercial Name', 'SSD Brand', 'Graphics', 'Audio']

# Function to apply hashing on a dataframe
def hash_features(df, n_features=10):
    hasher = FeatureHasher(n_features=n_features, input_type='string')
    df_as_dicts = df.astype(str).to_dict(orient='records')
    hashed_features = hasher.transform(df_as_dicts)
    return hashed_features.toarray()

# Create a hashing transformer
hashing_transformer = FunctionTransformer(hash_features, kw_args={'n_features': 10}, validate=False)

# ColumnTransformer for numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

# Combine numeric and hashed categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
    ],
    remainder='drop')  # Drop the categorical features for now

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Preprocess the categorical features using hashing separately
X_train_hashed = hashing_transformer.fit_transform(X_train[categorical_features])
X_test_hashed = hashing_transformer.transform(X_test[categorical_features])

# Preprocess the numeric features using the ColumnTransformer
X_train_preprocessed = preprocessor.fit_transform(X_train[numeric_features])
X_test_preprocessed = preprocessor.transform(X_test[numeric_features])

# Concatenate numeric and categorical features back together
X_train_combined = np.hstack((X_train_preprocessed, X_train_hashed))
X_test_combined = np.hstack((X_test_preprocessed, X_test_hashed))

# Train the model
model = XGBRegressor(objective='reg:squarederror')
model.fit(X_train_combined, y_train)

# Predictions
y_pred = model.predict(X_test_combined)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f'MSE: {mse}, MAE: {mae}, MAPE: {mape}')

# Plot test vs. predicted data points
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Measured vs. Predicted PCF')
plt.savefig('output/xgb_m_carbon_plot.png')
plt.clf()  # Clear the figure to prevent reusing labels and titles

# Save the model
dump(model, 'output/reg_model.joblib')

# SHAP explanation
explainer = shap.Explainer(model)
shap_values = explainer(X_train_combined)

# Generate feature names for hashed features
hashed_feature_names = ['hashed_feature_{}'.format(i) for i in range(10)]
# Combine numeric and hashed feature names
feature_names = numeric_features + hashed_feature_names

# Extract SHAP values for each feature and each sample
shap_values_array = shap_values.values  # This is a matrix of shape (# samples, # features)

# Calculate the mean absolute SHAP values for each feature across all samples
mean_abs_shap_values = np.abs(shap_values_array).mean(axis=0)

# Plotting with matplotlib
fig, ax = plt.subplots()
# Sort the feature indices based on mean absolute SHAP values
sorted_indices = np.argsort(mean_abs_shap_values)
# Use only the first four features (the non-hashed features)
top_indices = sorted_indices[-5:]  # Adjust this line to select the top features
# Create the bar plot
ax.barh(range(5), mean_abs_shap_values[top_indices], align='center', color='skyblue')
ax.set_yticks(range(5))
ax.set_yticklabels(np.array(feature_names)[top_indices])
ax.set_xlabel('mean(|SHAP value|) (average impact on model output magnitude)')
plt.tight_layout()
plt.savefig('output/shap_values_plot.png')
plt.clf()  # Clear the figure to prevent reusing labels and titles
