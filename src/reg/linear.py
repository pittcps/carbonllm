import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from joblib import dump
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('input/hp_combined_1.csv')

# Remove outliers where 'PCF' > 3500
df = df[df['PCF'] <= 3500]

# Selecting specified columns
columns = ['Commercial Name', 'Processor Cores', 'Memory', 'SSD', 'SSD Brand', 'Power', 'OS', 'Graphics', 'Audio', 'PCF']
df_selected = df[columns]

# Separating the target variable and features
X = df_selected.drop('PCF', axis=1)
y = df_selected['PCF']

# Defining numeric and categorical features
numeric_features = ['Processor Cores', 'Memory', 'SSD', 'Power']
categorical_features = ['Commercial Name', 'SSD Brand', 'OS', 'Graphics', 'Audio']

# Creating pipelines for both numeric and categorical preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combining preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Creating a pipeline that preprocesses the data and then trains a Linear Regression model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])
# model = Pipeline(steps=[('preprocessor', preprocessor),
#                         ('poly', PolynomialFeatures(degree=3, include_bias=False)),
#                         ('regressor', LinearRegression())])

# Splitting dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Training the model
model.fit(X_train, y_train)

# Predicting on test set
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

# Printing evaluation metrics
print(f'MSE: {mse}, MAE: {mae}, MAPE: {mape}')

# Plotting test vs predicted data points
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted PCF')
plt.savefig('out/carbon_plot.png')
plt.show()

# Saving the model
dump(model, 'out/reg_model.joblib')
