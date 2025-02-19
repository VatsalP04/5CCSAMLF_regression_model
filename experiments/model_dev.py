
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Import functions from data_preprocessing.py
from data_preprocessing import preprocess_data

# Import functions from feature_selection.py
from feature_selection import (
    feature_selection_correlation,
    low_variance_filter,
    feature_selection_mutual_info,
    apply_pca,
    rfecv_feature_selection,
    sfs_feature_selection
)

# ======================
# RANDOM FOREST
# ======================

def train_random_forest(X_train, y_train, X_val, y_val):
    """Train and evaluate a Random Forest model with hyperparameter tuning."""
    # Define hyperparameter search space
    param_dist = {
        "n_estimators": [250, 300, 350, 400],
        "max_depth": [10, 15, 20, 25, None],
        "min_samples_split": [2, 5, 10, 15, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", "log2"],
        "bootstrap": [True]
    }

    # Initialize Random Forest model
    rf_model = RandomForestRegressor(random_state=42)

    # Perform RandomizedSearchCV
    random_search = RandomizedSearchCV(
        rf_model,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=2
    )

    # Fit hyperparameter search on training data
    random_search.fit(X_train, y_train)

    # Get the best model
    best_params = random_search.best_params_
    final_model = random_search.best_estimator_

    # Evaluate model on validation set
    y_pred_final = final_model.predict(X_val)
    final_r2 = r2_score(y_val, y_pred_final)
    final_mse = mean_squared_error(y_val, y_pred_final)

    # Print results
    print("\nBest Hyperparameters for Random Forest:\n", best_params)
    print(f"Final R² Score: {final_r2:.4f}")
    print(f"Final MSE: {final_mse:.4f}")

    return final_model, best_params, final_r2, final_mse

# ======================
# XGBOOST
# ======================

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train and evaluate an XGBoost model with hyperparameter tuning."""
    # Define XGBoost hyperparameter grid
    xgb_param_dist = {
        "n_estimators": [200, 300, 400],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0, 0.1, 0.3],
        "lambda": [0.8, 1.0, 1.2],
        "alpha": [0, 0.1, 0.5]
    }

    # Initialize XGBoost model
    xgb_model = XGBRegressor(objective="reg:squarederror", random_state=42)

    # Hyperparameter tuning with RandomizedSearchCV
    xgb_random_search = RandomizedSearchCV(
        xgb_model,
        param_distributions=xgb_param_dist,
        n_iter=20,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        random_state=42,
        verbose=2
    )

    # Train model with hyperparameter tuning
    xgb_random_search.fit(X_train, y_train)

    # Get the best model
    best_xgb_params = xgb_random_search.best_params_
    final_xgb_model = xgb_random_search.best_estimator_

    # Evaluate XGBoost on validation set
    y_pred_xgb = final_xgb_model.predict(X_val)
    final_r2_xgb = r2_score(y_val, y_pred_xgb)
    final_mse_xgb = mean_squared_error(y_val, y_pred_xgb)

    # Print results
    print("\nBest Hyperparameters for XGBoost:\n", best_xgb_params)
    print(f"Final R² Score (XGBoost): {final_r2_xgb:.4f}")
    print(f"Final MSE (XGBoost): {final_mse_xgb:.4f}")

    return final_xgb_model, best_xgb_params, final_r2_xgb, final_mse_xgb

# ======================
# CATBOOST
# ======================

def train_catboost(X_train, y_train, X_val, y_val):
    """Train and evaluate a CatBoost model with hyperparameter tuning."""
    # Define hyperparameter grid
    param_grid = {
        'iterations': [500, 700, 1000],
        'learning_rate': [0.01, 0.02, 0.05],
        'depth': [4, 5, 6, 7],
        'l2_leaf_reg': [2, 3, 5, 7, 9],
        'colsample_bylevel': [0.8, 1.0],
        'subsample': [0.7, 0.8, 0.9],
        'bootstrap_type': ['Bayesian', 'Bernoulli']
    }

    # Initialize CatBoost model
    cat_model = CatBoostRegressor(loss_function='RMSE', random_state=42, verbose=False)

    # Hyperparameter tuning with RandomizedSearchCV
    cat_random_search = RandomizedSearchCV(
        cat_model,
        param_distributions=param_grid,
        n_iter=20,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=2
    )

    # Train model with hyperparameter tuning
    cat_random_search.fit(X_train, y_train)

    # Get the best model
    best_cat_params = cat_random_search.best_params_
    final_cat_model = cat_random_search.best_estimator_

    # Evaluate CatBoost on validation set
    y_pred_cat = final_cat_model.predict(X_val)
    final_r2_cat = r2_score(y_val, y_pred_cat)
    final_mse_cat = mean_squared_error(y_val, y_pred_cat)

    # Print results
    print("\nBest Hyperparameters for CatBoost:\n", best_cat_params)
    print(f"Final R² Score (CatBoost): {final_r2_cat:.4f}")
    print(f"Final MSE (CatBoost): {final_mse_cat:.4f}")

    return final_cat_model, best_cat_params, final_r2_cat, final_mse_cat

# ======================
# MAIN FUNCTION
# ======================

def main():
    # Preprocess data using the preprocess_data function
    file_path = "../CW1_train.csv"  # Update with the correct file path
    categorical_features = ['cut', 'color', 'clarity']
    save_path = "CW1_transformed.csv"  # Optional: Save the transformed data
    df_encoded = preprocess_data(file_path, categorical_features, save_path)

    # Prepare features and target
    target_col = "outcome"
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]

    # Train-test split (80-20)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Example feature sets (replace with actual selected features)
    feature_sets = {
        'SFS': ['depth', 'b3', 'b1', 'a1', 'a4', 'a3', 'b6'],
        'RFECV': ['depth', 'b3', 'b1', 'a1', 'a4', 'a3', 'a2', 'b5', 'b10', 'b9', 'a7', 'a10', 'price'],
        'RFECV_More_Features': [
            'depth', 'table', 'a1', 'a2', 'a3', 'a4', 'a5', 'b1', 'b2', 'b3', 'b4',
            'b5', 'a6', 'a7', 'a8', 'a9', 'a10', 'b6', 'b7', 'b8', 'b9', 'b10',
            'cut_Good', 'cut_Ideal', 'cut_Premium', 'cut_Very Good', 'color_E',
            'color_F', 'color_G', 'color_H', 'color_I', 'color_J', 'clarity_SI1',
            'clarity_SI2', 'clarity_VS1', 'clarity_VS2', 'clarity_VVS1',
            'clarity_VVS2', 'volume', 'log_price', 'log_carat'
        ]
    }

    # Train and evaluate models for each feature set
    for method, features in feature_sets.items():
        print(f"\nEvaluating {method} Features:")
        X_selected = X[features]

        # Train-test split for selected features
        X_train_selected, X_val_selected, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42)

        # Train and evaluate Random Forest
        print("\nRandom Forest Results:")
        train_random_forest(X_train_selected, y_train, X_val_selected, y_val)

        # Train and evaluate XGBoost
        print("\nXGBoost Results:")
        train_xgboost(X_train_selected, y_train, X_val_selected, y_val)

        # Train and evaluate CatBoost
        print("\nCatBoost Results:")
        train_catboost(X_train_selected, y_train, X_val_selected, y_val)

if __name__ == "__main__":
    main()
