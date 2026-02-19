
# Importing Libraries
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score


#Load Dataset
housing = fetch_california_housing()

X = housing.data          
y = housing.target      

# Convert to DataFrame
df = pd.DataFrame(X, columns=housing.feature_names)
df["Price"] = y

print("Dataset Shape:", df.shape)
print(df.head())


# Data Preprocessing 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Split Dataset fot testing and training
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Training size:", X_train.shape)
print("Testing size:", X_test.shape)


# linear regression model

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("\n----- Linear Regression -----")
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R2 Score:", r2_score(y_test, y_pred_lr))


# decision tree model

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("\n Decision Tree")
print("MSE:", mean_squared_error(y_test, y_pred_dt))
print("R2 Score:", r2_score(y_test, y_pred_dt))


# 6️⃣ Predict New Data
sample = X_test[0].reshape(1, -1)

print("\nActual Price:", y_test[0])
print("Linear Regression :", lr.predict(sample)[0])
print("Decision Tree :", dt.predict(sample)[0])
