
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor

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
# INITIALIZE MODELS
# ======================

def initialize_models():
    """Initialize a dictionary of regression models."""
    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
        "XGBoost": XGBRegressor(objective="reg:squarederror", random_state=42),
        "Linear Regression": LinearRegression(),
        "CatBoost": CatBoostRegressor(verbose=False)
    }
    return models

# ======================
# CROSS-VALIDATION
# ======================

def evaluate_models_cv(models, X, y, cv=5, scoring='r2'):
    """Evaluate models using cross-validation and print R² scores."""
    results = {}
    for name, model in models.items():
        score = cross_val_score(model, X, y, cv=cv, scoring=scoring).mean()
        results[name] = score
        print(f"{name} R² Score: {score:.4f}")
    return results

# ======================
# FEATURE IMPORTANCE VISUALIZATION
# ======================

def plot_feature_importances(model, feature_names, top_n=10):
    """Plot the top N feature importances from a model."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importances = feature_importances.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importances['Feature'][:top_n], feature_importances['Importance'][:top_n])
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.show()
    else:
        print("Model does not support feature importances.")

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
    X_full = df_encoded.drop(columns=['outcome'])
    y = df_encoded['outcome']
    
    # Initialize models
    models = initialize_models()
    
    # Evaluate models on full feature set
    print("\nEvaluating Models on Full Feature Set:")
    evaluate_models_cv(models, X_full, y)
    
    # Example feature sets (replace with actual selected features)
    feature_sets = {
        'SFS': ['depth', 'b3', 'b1', 'a1', 'a4', 'a3', 'b6'],
        'RFECV': ['depth', 'b3', 'b1', 'a1', 'a4', 'a3', 'a2', 'b5', 'b10', 'b9', 'a7', 'a10', 'log_price'],
        'RFECV_More_Features': [
            'depth', 'table', 'a1', 'a2', 'a3', 'a4', 'a5', 'b1', 'b2', 'b3', 'b4',
            'b5', 'a6', 'a7', 'a8', 'a9', 'a10', 'b6', 'b7', 'b8', 'b9', 'b10',
            'cut_Good', 'cut_Ideal', 'cut_Premium', 'cut_Very Good', 'color_E',
            'color_F', 'color_G', 'color_H', 'color_I', 'color_J', 'clarity_SI1',
            'clarity_SI2', 'clarity_VS1', 'clarity_VS2', 'clarity_VVS1',
            'clarity_VVS2', 'volume', 'log_price', 'log_carat'
        ]
    }
    
    # Compare feature selection methods
    for method, features in feature_sets.items():
        print(f"\nEvaluating {method} Features:")
        df_selected = df_encoded[features]
        evaluate_models_cv(models, df_selected, y)
    
    # # Example of PCA usage
    # X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)
    # X_train_pca, X_test_pca = apply_pca(X_train, X_test, n_components=10)
    
    # # Train and evaluate a model with PCA
    # model = RandomForestRegressor(n_estimators=50, random_state=42)
    # model.fit(X_train_pca, y_train)
    # y_pred_pca = model.predict(X_test_pca)
    # pca_r2 = r2_score(y_test, y_pred_pca)
    # print(f"\nPCA-Reduced R² Score: {pca_r2:.4f}")
    
    # # Plot feature importances
    # plot_feature_importances(model, feature_names=X_full.columns, top_n=10)

if __name__ == "__main__":
    main()