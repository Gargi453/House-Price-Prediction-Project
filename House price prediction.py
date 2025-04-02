# Importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# Fetching California Housing dataset
house_price_dataset = sklearn.datasets.fetch_california_housing()

# Loading dataset into pandas DataFrame
house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns=house_price_dataset.feature_names)

# Adding the target column (MedianHouseValue)
house_price_dataframe['MedianHouseValue'] = house_price_dataset.target

# Display first few rows
print(house_price_dataframe.head())

# Checking number of rows and columns
print("number of rows and columns:", house_price_dataframe.shape)

# Checking for missing values
print("Missing Values:\n", house_price_dataframe.isnull().sum())

# Statistical summary
print(house_price_dataframe.describe())

# Understanding correlation
correlation = house_price_dataframe.corr()

# Constructing heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Reds')
plt.title("Feature Correlation Heatmap")
plt.show()

# Splitting data into features (X) and target (y)
X = house_price_dataframe.drop(['MedianHouseValue'], axis=1)  # Features
y = house_price_dataframe['MedianHouseValue']  # Target variable

# Splitting data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Model training using XGBoost Regressor
model = XGBRegressor()
model.fit(X_train, y_train)

# Predictions on training data
train_predictions = model.predict(X_train)

# Evaluating training data predictions
train_r2_score = metrics.r2_score(y_train, train_predictions)
train_mae = metrics.mean_absolute_error(y_train, train_predictions)

print("Training Data Evaluation:")
print(f"R squared Score: {train_r2_score:.4f}")
print(f"Mean Absolute Error: {train_mae:.4f}")

# Predictions on test data
test_predictions = model.predict(X_test)

# Evaluating test data predictions
test_r2_score = metrics.r2_score(y_test, test_predictions)
test_mae = metrics.mean_absolute_error(y_test, test_predictions)

print("\nTest Data Evaluation:")
print(f"R squared Score: {test_r2_score:.4f}")
print(f"Mean Absolute Error: {test_mae:.4f}")

# Visualizing actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, test_predictions, alpha=0.5, color="blue")
plt.xlabel("Actual Median House Value ($100,000s)")
plt.ylabel("Predicted Median House Value ($100,000s)")
plt.title("Actual vs. Predicted House Prices")
plt.show()



