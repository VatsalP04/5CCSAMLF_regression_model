import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Load the training dataset
train_file_path = "../CW1_train.csv"  # Change to your file path if needed
train_data = pd.read_csv(train_file_path)

# Define the selected features
selected_features = ['depth', 'table', 'a1', 'a2', 'a3', 'a4', 'a5', 'b1', 'b2', 'b3', 'b4',
                     'b5', 'a6', 'a7', 'a8', 'a9', 'a10', 'b6', 'b7', 'b8', 'b9', 'b10',
                     'cut_Good', 'color_E', 'color_G', 'color_H', 'color_I', 'color_J', 
                     'clarity_SI1', 'clarity_VS1', 'clarity_VVS2', 'volume', 'log_price', 'log_carat']

# One-hot encode categorical features
categorical_features = ['cut', 'color', 'clarity']
train_data = pd.get_dummies(train_data, columns=categorical_features, drop_first=True)

# Create 'volume' feature
train_data['volume'] = train_data['x'] * train_data['y'] * train_data['z']

# Apply log transformation
train_data['log_price'] = np.log1p(train_data['price'])
train_data['log_carat'] = np.log1p(train_data['carat'])

# Drop original features that were transformed
train_data.drop(columns=['x', 'y', 'z', 'price', 'carat'], inplace=True)

# Extract training features and target
X_trn = train_data[selected_features]
y_trn = train_data['outcome']

# Initialize CatBoost model with best parameters
model = CatBoostRegressor(
    subsample=0.7,
    learning_rate=0.05,
    l2_leaf_reg=9,
    iterations=500,
    depth=4,
    colsample_bylevel=0.8,
    bootstrap_type='Bernoulli',
    loss_function='RMSE',
    verbose=200  # Adjust verbosity if needed
)


# model = RandomForestRegressor()
# model = LinearRegression()


# Perform 5-fold cross-validation
cv_r2_scores = cross_val_score(model, X_trn, y_trn, scoring='r2', cv=5)

# Compute mean cross-validation R²
mean_cv_r2 = cv_r2_scores.mean()

# Print results
print(f"Cross-Validation R² Scores: {cv_r2_scores}")
print(f"Mean Cross-Validation R²: {mean_cv_r2:.4f}")
