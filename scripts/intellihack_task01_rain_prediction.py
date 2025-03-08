#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/KaveenAmarasekara/Intellihack_RedHawk_Task01/blob/main/notebooks/intellihack_task01_rain_prediction.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Weather Forecasting for Smart Agriculture
# This project aims to predict whether it will rain or not based on historical weather data. The dataset includes features like temperature, humidity, and wind speed.

# In[1]:


# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


# In[ ]:


# Step 2: Load the Dataset
url = "https://raw.githubusercontent.com/KaveenAmarasekara/Intellihack_RedHawk_Task01/refs/heads/main/data/weather_data.csv"

#df = pd.read_csv('file') #you can replace this with your local file or use the url
df = pd.read_csv(url)
df.head()


# In[ ]:


# Step 3: Data Preprocessing
# Check for missing values
print(df.isnull().sum())


# In[ ]:


# # Fill missing values with the mean
# df.fillna(df.mean(), inplace=True)


# In[ ]:


print(df.dtypes)


# In[ ]:


df['date'] = pd.to_datetime(df['date'])


# In[ ]:


df['rain_or_not'] = df['rain_or_not'].map({'Rain': 1, 'No Rain': 0})


# In[ ]:


#since 'date' and 'rain_or_not' were non-numerics we had to change them to a relavant type to refill null values...


# In[ ]:


# Fill missing values with the mean (retry)
df.fillna(df.mean(), inplace=True)


# In[ ]:


print(df.dtypes)


# In[ ]:


df.head()


# In[ ]:


# Check for missing values
print(df.isnull().sum())


# In[ ]:


## no null values on second checking


# In[ ]:


# Step 1: Identify numeric columns
numeric_columns = df.select_dtypes(include=['int', 'float']).columns
print("Numeric columns:", numeric_columns)


# In[ ]:


## now checking for any negative values in above columns


# In[ ]:


for column in numeric_columns:
    negative_values = df[column][df[column] < 0]  # Filter rows with negative values
    if not negative_values.empty:
        print(f"\nColumn '{column}' has {len(negative_values)} negative values:")
        print(negative_values)


# In[ ]:


# seems there is no negative values.


# In[ ]:


# Now we extract day of the week and month from date and make separate columns for future use if needed
df['date'] = pd.to_datetime(df['date'])
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

# Drop
df.drop('date', axis=1, inplace=True)


# In[ ]:


print(df.dtypes)
df.head()


# In[ ]:


print(df.isnull().sum())


# In[ ]:


# no errors with new columns


# In[ ]:


# Normalize
scaler = StandardScaler()
df[['avg_temperature', 'humidity', 'avg_wind_speed', 'cloud_cover', 'pressure']] = scaler.fit_transform(df[['avg_temperature', 'humidity', 'avg_wind_speed', 'cloud_cover', 'pressure']])


# In[ ]:


# Step 4: (EDA) >> plot graphs

# Distribution of avg_temperature
plt.figure(figsize = (5, 5))
sns.histplot(df['avg_temperature'], kde=True) # Univariate Analysis
plt.title('Distribution of Average Temperature')
plt.tight_layout()
plt.show()

# Humidity vs Rain
plt.figure(figsize = (5, 3))
sns.boxplot(x='rain_or_not', y='humidity', data=df) # Bivariate Analysis
plt.title('Humidity vs Rain')
plt.tight_layout()
plt.show()

# Correlation Analysis
plt.figure(figsize = (12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()


# In[ ]:


# Split the data into features (X) and target (y)
X = df.drop('rain_or_not', axis=1)
y = df['rain_or_not']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


# Train a Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)


# In[ ]:


# Evaluate Logistic Regression
print("Logistic Regression Metrics:\n")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg)}")
print(f"Precision: {precision_score(y_test, y_pred_log_reg)}")
print(f"Recall: {recall_score(y_test, y_pred_log_reg)}")
print(f"F1-Score: {f1_score(y_test, y_pred_log_reg)}")
print(f"ROC-AUC: {roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])}")


# In[ ]:


# Train a Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


# In[ ]:


# Evaluate Random Forest
print("\nRandom Forest Metrics:\n")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(f"Precision: {precision_score(y_test, y_pred_rf)}")
print(f"Recall: {recall_score(y_test, y_pred_rf)}")
print(f"F1-Score: {f1_score(y_test, y_pred_rf)}")
print(f"ROC-AUC: {roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])}")


# In[ ]:


# Step 6: Model Optimization
# Hyperparameter Tuning for Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("\nBest Parameters for Random Forest:", grid_search.best_params_)


# In[ ]:


# Train the optimized model
best_rf = grid_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)


# In[ ]:


# Evaluate the optimized model
print("\nOptimized Random Forest Metrics:\n")
print(f"Accuracy: {accuracy_score(y_test, y_pred_best_rf)}")
print(f"Precision: {precision_score(y_test, y_pred_best_rf)}")
print(f"Recall: {recall_score(y_test, y_pred_best_rf)}")
print(f"F1-Score: {f1_score(y_test, y_pred_best_rf)}")
print(f"ROC-AUC: {roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1])}")


# Hard coded future dataset

# In[ ]:


# Step 7: Final Output
# Generate predictions for the next 21 days (example)
future_data = pd.DataFrame({
    'avg_temperature': [20, 22, 19, 30],
    'humidity': [60, 65, 70, 40],
    'avg_wind_speed': [10, 12, 11, 10],
    'cloud_cover': [50, 55, 60, 45],
    'pressure': [1010, 1012, 1011, 1010],
    'day_of_week': [1, 2, 3, 2],
    'month': [6, 6, 6, 1]
})

# Scale future data
future_data[['avg_temperature', 'humidity', 'avg_wind_speed', 'cloud_cover', 'pressure']] = scaler.transform(future_data[['avg_temperature', 'humidity', 'avg_wind_speed', 'cloud_cover', 'pressure']])

# Predict probabilities
future_predictions = best_rf.predict_proba(future_data)[:, 1]
print("\nRain Probability for Future Data:")
print(future_predictions)

# Save predictions to a CSV file
pd.DataFrame(future_predictions, columns=['rain_probability']).to_csv('future_predictions.csv', index=False)


# Handle a future data set (from a file)

# In[ ]:


# Step 1: Load the future data from CSV
url = "https://raw.githubusercontent.com/KaveenAmarasekara/Intellihack_RedHawk_Task01/refs/heads/main/data/future_data.csv"
# future_data = pd.read_csv('file') #you can replace this with your local file or use the url
future_data = pd.read_csv(url)
future_data.head()


# In[ ]:


print(future_data.isnull().sum())


# In[ ]:


print(future_data.dtypes,"\n")
future_data.head()
future_data['date'] = pd.to_datetime(future_data['date'])

# Step 2: Handle missing values (if any)
# Fill missing values with the mean of each column
future_data.fillna(future_data.mean(), inplace=True)

# Step 3: Handle incorrect entries (e.g., negative values)
# Replace negative values with the column mean
for column in future_data.select_dtypes(include=['int', 'float']).columns:
    future_data[column] = future_data[column].apply(lambda x: x if x >= 0 else future_data[column].mean())

future_data['date'] = pd.to_datetime(future_data['date'])

future_data['day_of_week'] = future_data['date'].dt.dayofweek
future_data['month'] = future_data['date'].dt.month

# Drop
future_data.drop('date', axis=1, inplace=True)
print(future_data.dtypes)
future_data.head()



# In[ ]:


# Step 4: Scale the future data (if scaling was applied to training data)
# Load the scaler used for training data (or fit a new one)
scaler = StandardScaler()
# Assuming the training data was scaled, fit the scaler on the training data
# For example, if X_train was used for scaling:
# scaler.fit(X_train)

# Scale the future data

scaler = StandardScaler()
future_data[['avg_temperature', 'humidity', 'avg_wind_speed', 'cloud_cover', 'pressure']] = scaler.fit_transform(future_data[['avg_temperature', 'humidity', 'avg_wind_speed', 'cloud_cover', 'pressure']])


# Convert back to DataFrame (optional)
future_data_scaled = pd.DataFrame(future_data, columns=future_data.columns)

# Step 5: Verify the preprocessed future data
print("Preprocessed Future Data:")
future_data_scaled.head()


# In[ ]:


# Assuming you have a trained model (e.g., Random Forest)
predictions = best_rf.predict_proba(future_data_scaled)[:, 1]  # Probability of rain

# Save predictions to a CSV file
pd.DataFrame(predictions, columns=['rain_probability']).to_csv('future_predictions.csv', index=False)

print("Rain Probabilities for Future Data:")
print(predictions)

