import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("--- Starting California Housing Price Prediction ---")

# --- 1. Dataset Loading ---
print("\n### 1. Dataset Loading ###")
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name='MedianHouseValue')

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print("\nFirst 5 rows of features:")
print(X.head())
print("\nTarget variable statistics:")
print(y.describe())

# --- 2. Exploratory Data Analysis (EDA) ---
print("\n### 2. Exploratory Data Analysis (EDA) ###")

# Check for missing values
print("\nMissing values in features:")
print(X.isnull().sum())
print("\nMissing values in target:")
print(y.isnull().sum())

# Visualize the distribution of the target variable
plt.figure(figsize=(10, 6))
sns.histplot(y, kde=True, bins=50)
plt.title('Distribution of Median House Value')
plt.xlabel('Median House Value ($100,000s)')
plt.ylabel('Frequency')
plt.show()

# Visualize the correlation matrix
plt.figure(figsize=(12, 10))
correlation_matrix = pd.concat([X, y], axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features and Target')
plt.show()

# --- 3. Data Preprocessing ---
print("\n### 3. Data Preprocessing ###")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Testing set shape: {X_test.shape}, {y_test.shape}")

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames for easier feature selection
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
print("\nFeatures scaled successfully.")
print("First 5 rows of scaled training features:")
print(X_train_scaled.head())

# --- 4. Feature Selection ---
print("\n### 4. Feature Selection ###")

# Use SelectKBest with f_regression to score features
selector = SelectKBest(score_func=f_regression, k='all')
selector.fit(X_train_scaled, y_train)

feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'Score': selector.scores_,
    'P_Value': selector.pvalues_
})
feature_scores = feature_scores.sort_values(by='Score', ascending=False)

print("\nFeature importance scores (based on F-regression):")
print(feature_scores)

# Select the top 6 features based on scores (this number can be adjusted)
k_best_features = 6
selected_features_names = feature_scores.head(k_best_features)['Feature'].tolist()
print(f"\nSelected {k_best_features} features for modeling: {selected_features_names}")

X_train_selected = X_train_scaled[selected_features_names]
X_test_selected = X_test_scaled[selected_features_names]

print(f"\nShape of X_train after feature selection: {X_train_selected.shape}")
print(f"Shape of X_test after feature selection: {X_test_selected.shape}")

# --- 5. Model Selection and Training ---
print("\n### 5. Model Selection and Training ###")

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(random_state=42, n_jobs=-1),
    'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42)
}

model_performance = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    model_performance[name] = {
        'MSE': mse,
        'RMSE': rmse,
        'R2 Score': r2,
        'MAE': mae
    }

    print(f"{name} Performance:")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  R-squared (R2): {r2:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")

# Display a summary of all model performances
print("\n--- Summary of Initial Model Performances ---")
performance_df = pd.DataFrame(model_performance).T
print(performance_df)

# --- 6. Hyperparameter Tuning (for Gradient Boosting Regressor) ---
print("\n### 6. Hyperparameter Tuning for Gradient Boosting Regressor ###")

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

gbm_model = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(estimator=gbm_model, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')

grid_search.fit(X_train_selected, y_train)

print(f"\nBest parameters found: {grid_search.best_params_}")
print(f"Best cross-validation MSE (negative because scoring is 'neg_mean_squared_error'): {-grid_search.best_score_:.4f}")

# Evaluate the best model on the test set
best_gbm_model = grid_search.best_estimator_
y_pred_tuned = best_gbm_model.predict(X_test_selected)

mse_tuned = mean_squared_error(y_test, y_pred_tuned)
rmse_tuned = np.sqrt(mse_tuned)
r2_tuned = r2_score(y_test, y_pred_tuned)
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)

print("\nTuned Gradient Boosting Regressor Performance on Test Set:")
print(f"  Mean Squared Error (MSE): {mse_tuned:.4f}")
print(f"  Root Mean Squared Error (RMSE): {rmse_tuned:.4f}")
print(f"  R-squared (R2): {r2_tuned:.4f}")
print(f"  Mean Absolute Error (MAE): {mae_tuned:.4f}")

# Visualize actual vs. predicted values for the best model
plt.figure(figsize=(10, 7))
sns.regplot(x=y_test, y=y_pred_tuned, scatter_kws={'alpha':0.3, 'color':'blue'})
plt.xlabel("Actual Median House Value ($100,000s)")
plt.ylabel("Predicted Median House Value ($100,000s)")
plt.title("Actual vs. Predicted Median House Values (Tuned Gradient Boosting)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) # y=x line
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print("\n--- Model Building and Evaluation Complete ---")
