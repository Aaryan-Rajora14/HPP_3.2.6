import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('delhi_houses (1).csv')

print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# Check for data issues
print("\nUnique values in categorical columns:")
for col in df.select_dtypes(include=['object']).columns:
    print(f"{col}: {df[col].unique()}")


# Fix the typo in 'Semi-Furnsihed'
df['furnishingstatus'] = df['furnishingstatus'].replace('Semi-Furnsihed', 'Semi-Furnished')

# Handle categorical variables
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                   'airconditioning', 'prefarea', 'furnishingstatus']

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Prepare features and target
X = df.drop('price', axis=1)
y = df['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set shape: {X_train_scaled.shape}")
print(f"Test set shape: {X_test_scaled.shape}")


# Initial Gradient Boosting Model
gb_model = GradientBoostingRegressor(random_state=42)

# Fit initial model
gb_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_train = gb_model.predict(X_train_scaled)
y_pred_test = gb_model.predict(X_test_scaled)

# Calculate metrics for initial model
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("Initial Gradient Boosting Performance:")
print(f"Training R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Training RMSE: {train_rmse:,.2f}")
print(f"Test RMSE: {test_rmse:,.2f}")

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.15],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'subsample': [0.8, 0.9, 1.0]
}

# Using RandomizedSearchCV for faster tuning
from sklearn.model_selection import RandomizedSearchCV

gb_tuned = GradientBoostingRegressor(random_state=42)

random_search = RandomizedSearchCV(
    gb_tuned, 
    param_grid, 
    n_iter=50, 
    cv=5, 
    scoring='r2', 
    n_jobs=-1, 
    random_state=42
)

print("\nPerforming hyperparameter tuning...")
random_search.fit(X_train_scaled, y_train)

# Best model from tuning
best_gb_model = random_search.best_estimator_

# Predictions with tuned model
y_pred_train_tuned = best_gb_model.predict(X_train_scaled)
y_pred_test_tuned = best_gb_model.predict(X_test_scaled)

# Calculate metrics for tuned model
train_r2_tuned = r2_score(y_train, y_pred_train_tuned)
test_r2_tuned = r2_score(y_test, y_pred_test_tuned)
train_rmse_tuned = np.sqrt(mean_squared_error(y_train, y_pred_train_tuned))
test_rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_test_tuned))

print("\n" + "="*50)
print("TUNED GRADIENT BOOSTING PERFORMANCE:")
print("="*50)
print(f"Best Parameters: {random_search.best_params_}")
print(f"Training R²: {train_r2_tuned:.4f}")
print(f"Test R²: {test_r2_tuned:.4f}")
print(f"Training RMSE: {train_rmse_tuned:,.2f}")
print(f"Test RMSE: {test_rmse_tuned:,.2f}")

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_gb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Create a dictionary to save all necessary objects
model_package = {
    'model': best_gb_model,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'feature_names': X.columns.tolist(),
    'performance': {
        'train_r2': train_r2_tuned,
        'test_r2': test_r2_tuned,
        'train_rmse': train_rmse_tuned,
        'test_rmse': test_rmse_tuned
    }
}

# Save the model package
joblib.dump(model_package, 'HPP_Model_3.2.pkl')

print("Model saved successfully as HPP_Model_3.2.pkl")

# Verification - Load and test the saved model
loaded_package = joblib.load('HPP_Model_3.2.pkl')
loaded_model = loaded_package['model']

# Quick test
test_pred = loaded_model.predict(X_test_scaled)
test_r2_loaded = r2_score(y_test, test_pred)
print(f"Loaded model test R²: {test_r2_loaded:.4f}")
