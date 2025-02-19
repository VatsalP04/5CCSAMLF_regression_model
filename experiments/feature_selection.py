import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression, RFECV, SequentialFeatureSelector
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ======================
# FEATURE SELECTION - CORRELATION
# ======================

def feature_selection_correlation(df, outcome_col='outcome', threshold=0.7):
    """
    Identify correlated features and their relationship with the outcome.
    Returns correlation matrix and correlated features.
    """
    def correlation(dataset, threshold):
        col_corr = set()
        corr_matrix = dataset.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    colname = corr_matrix.columns[i]
                    col_corr.add(colname)
        return col_corr

    corr_features = correlation(df, threshold)
    
    # Get correlations with outcome
    outcome_corr = df.corr()[outcome_col].sort_values(ascending=False)
    
    # Get correlation matrix for correlated features
    if corr_features:
        corr_matrix = df[list(corr_features) + [outcome_col]].corr()
        print("Correlation with outcome:")
        print(corr_matrix[outcome_col].sort_values(ascending=False))
        print("\nCorrelation among correlated features:")
        print(corr_matrix.loc[list(corr_features), list(corr_features)])
    else:
        corr_matrix = pd.DataFrame()
    
    return corr_matrix, corr_features

# ======================
# LOW VARIANCE FEATURE FILTERING
# ======================

def low_variance_filter(df, outcome_col='outcome', threshold=0.01):
    """Filter features with low variance, returns filtered data and removed features"""
    selector = VarianceThreshold(threshold=threshold)
    X = df.drop(columns=[outcome_col])
    selector.fit(X)
    
    selected_features = X.columns[selector.get_support()]
    removed_features = set(X.columns) - set(selected_features)
    
    print("Removed Low-Variance Features:", removed_features)
    return df[selected_features.tolist() + [outcome_col]], removed_features

# ======================
# MUTUAL INFORMATION FEATURE SELECTION
# ======================

def feature_selection_mutual_info(df, outcome_col='outcome', threshold=0.001):
    """Select features using Mutual Information (Information Gain)"""
    # Identify categorical features
    categorical_features = [col for col in df.columns 
                          if any(k in col for k in ['cut', 'color', 'clarity'])]
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[categorical_features])
    
    # Calculate mutual information
    info_gain = mutual_info_regression(X_scaled, df[outcome_col])
    
    # Create results dataframe
    ig_df = pd.DataFrame({'Feature': categorical_features, 'IG': info_gain})
    ig_df = ig_df.sort_values('IG', ascending=False)
    
    # Identify low IG features
    low_ig_features = ig_df[ig_df['IG'] < threshold]['Feature'].tolist()
    df_reduced = df.drop(columns=low_ig_features)
    
    print(f"Removed Features: {low_ig_features}")
    return df_reduced, low_ig_features

# ======================
# DIMENSIONALITY REDUCTION (PCA)
# ======================

def apply_pca(df, feature_groups, n_components=5):
    """Apply PCA to specified feature groups"""
    df_pca = df.copy()
    
    for group in feature_groups:
        group_features = [f'{group}{i}' for i in range(1, 11)]
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(df_pca[group_features])
        component_names = [f'PCA_{group.upper()}_{i+1}' for i in range(n_components)]
        df_pca[component_names] = components
        df_pca.drop(columns=group_features, inplace=True)
    
    return df_pca

# ======================
# RFECV FEATURE SELECTION (WRAPPER)
# ======================

def rfecv_feature_selection(file_path, outcome_col='outcome', test_size=0.2, random_state=42):
    """Perform recursive feature elimination with cross-validation"""
    df = pd.read_csv(file_path)
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # One-hot encode categorical features
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_encoded = pd.DataFrame(encoder.fit_transform(df[categorical_cols]))
    X_encoded.columns = encoder.get_feature_names_out(categorical_cols)
    
    # Prepare final dataset
    X = df.drop(columns=categorical_cols.tolist() + [outcome_col])
    X_final = pd.concat([X.reset_index(drop=True), X_encoded], axis=1)
    y = df[outcome_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=test_size, random_state=random_state)
    
    # Initialize and fit RFECV
    model = RandomForestRegressor(random_state=random_state)
    rfecv = RFECV(estimator=model, step=1, cv=5, scoring='r2')
    rfecv.fit(X_train, y_train)
    
    # Get results
    selected_features = X_final.columns[rfecv.support_]
    model.fit(X_train[selected_features], y_train)
    y_pred = model.predict(X_test[selected_features])
    best_r2 = r2_score(y_test, y_pred)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), 
             rfecv.cv_results_['mean_test_score'], marker='o')
    plt.xlabel('Number of Selected Features')
    plt.ylabel('Cross-Validated R2 Score')
    plt.title('Feature Selection with RFECV')
    plt.grid(True)
    plt.show()
    
    # Create feature selection dictionary
    feature_selection_dict = {
        n_features: X_train.columns[np.argsort(rfecv.ranking_)[:n_features]].tolist()
        for n_features in range(1, len(X_train.columns) + 1)
    }
    
    return selected_features, best_r2, feature_selection_dict

# ======================
# FEATURE SELECTION - SFS
# ======================

def sfs_feature_selection(file_path, categorical_cols=None, test_size=0.2, random_state=42, top_n=15, k_features=5):
    """Perform Sequential Feature Selection"""
    df = pd.read_csv(file_path)
    
    # Handle categorical features
    categorical_cols = categorical_cols or ['cut', 'color', 'clarity']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Prepare data
    X = df_encoded.drop(columns=['outcome'])
    y = df_encoded['outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Get initial feature importance
    model = RandomForestRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    
    # Select top features
    importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    top_features = importance_df.nlargest(top_n, 'Importance')['Feature'].tolist()
    
    # Perform SFS
    sfs = SequentialFeatureSelector(model, 
                                   k_features=k_features,
                                   forward=True,
                                   scoring='r2',
                                   cv=3,
                                   n_jobs=-1)
    sfs.fit(X_train[top_features], y_train)
    
    return top_features, sfs.k_feature_names_

# ======================
# MAIN FUNCTION
# ======================

def main():
    # Example usage flow
    df = pd.read_csv("CW1_transformed.csv")
    
    # # Correlation-based selection
    # corr_matrix, corr_features = feature_selection_correlation(df)
    
    # # Low variance filtering
    # df_low_var, removed_low_var = low_variance_filter(df)
    
    # # Mutual information selection
    # df_mutual_info, removed_ig = feature_selection_mutual_info(df)
    
    # # PCA
    # pca_df = apply_pca(df, feature_groups=['a', 'b'])
    
    # # RFECV
    # rfecv_features, r2_score, fs_dict = rfecv_feature_selection("CW1_Reduced.csv")
    
    # # SFS
    # top_features, sfs_features = sfs_feature_selection("CW1_transformed.csv")

if __name__ == "__main__":
    main()