import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# Set seed for reproducibility
np.random.seed(123)

# Load datasets
train_data = pd.read_csv('CW1_train.csv')
test_data = pd.read_csv('CW1_test.csv')  # No true outcomes included

# Identify categorical columns
categorical_features = ['cut', 'color', 'clarity']

# One-hot encode categorical features
train_data = pd.get_dummies(train_data, columns=categorical_features, drop_first=True)
test_data = pd.get_dummies(test_data, columns=categorical_features, drop_first=True)

# Create 'volume' feature
train_data['volume'] = train_data['x'] * train_data['y'] * train_data['z']
test_data['volume'] = test_data['x'] * test_data['y'] * test_data['z']

# Apply log transformation
train_data['log_price'] = np.log1p(train_data['price'])
train_data['log_carat'] = np.log1p(train_data['carat'])
test_data['log_price'] = np.log1p(test_data['price'])
test_data['log_carat'] = np.log1p(test_data['carat'])

# Drop original features that were transformed
train_data.drop(columns=['x', 'y', 'z', 'price', 'carat'], inplace=True)
test_data.drop(columns=['x', 'y', 'z', 'price', 'carat'], inplace=True)

# Define selected features (from best feature selection)
selected_features = ['depth', 'table', 'a1', 'a2', 'a3', 'a4', 'a5', 'b1', 'b2', 'b3', 'b4',
                     'b5', 'a6', 'a7', 'a8', 'a9', 'a10', 'b6', 'b7', 'b8', 'b9', 'b10',
                     'cut_Good', 'color_E', 'color_G', 'color_H', 'color_I', 'color_J', 
                     'clarity_SI1', 'clarity_VS1', 'clarity_VVS2', 'volume', 'log_price', 'log_carat']

# Ensure test set has the same feature columns as training
test_data = test_data.reindex(columns=train_data.columns, fill_value=0)

# Extract features & target
X_trn = train_data[selected_features]
y_trn = train_data['outcome']
X_tst = test_data[selected_features]  # No y_tst (unknown outcomes)

# Initialize and train CatBoost model with best parameters
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

model.fit(X_trn, y_trn)

# Generate predictions
yhat_cb = model.predict(X_tst)

# Save predictions in required format
submission = pd.DataFrame({'yhat': yhat_cb})
submission.to_csv('CW1_submission_K22015880.csv', index=False)  # Replace KNUMBER with your ID

