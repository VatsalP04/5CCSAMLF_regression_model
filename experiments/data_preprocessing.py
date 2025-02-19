# data_preprocessing.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

def load_data(file_path):
    """Load the dataset from the specified file path."""
    df = pd.read_csv(file_path)
    return df

def display_basic_info(df):
    """Display basic information about the dataset."""
    df.info()
    return df.head()

def detect_outliers(df, threshold=3):
    """Detect outliers in the numerical columns of the dataset."""
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
    outliers = np.where(z_scores > threshold)
    print("Outliers detected at positions:", outliers)
    return outliers

def visualize_outliers(df):
    """Visualize outliers using a boxplot."""
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=df.select_dtypes(include=[np.number]))
    plt.xticks(rotation=90)
    plt.title("Boxplot to visualize outliers")
    plt.show()

def encode_categorical_features(df, categorical_features):
    """Encode categorical features using one-hot encoding."""
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    return df_encoded

def feature_engineering(df):
    """Perform feature engineering by creating new features and transforming them."""
    df['volume'] = df['x'] * df['y'] * df['z']
    corr_matrix = df[['x', 'y', 'z', 'volume']].corr()
    print(corr_matrix)
    df.drop(columns=['x', 'y', 'z'], inplace=True)

    # Log transformation on skewed features
    df['log_price'] = np.log1p(df['price'])
    df['log_carat'] = np.log1p(df['carat'])

    # Drop original 'price' and 'carat' as their log-transformed versions are used
    df.drop(columns=['price', 'carat'], inplace=True)

    return df

def save_transformed_data(df, file_path):
    """Save the transformed dataset to a CSV file."""
    df.to_csv(file_path, index=False)

def preprocess_data(file_path, categorical_features, save_path=None):
    """
    Perform all preprocessing steps: load data, detect outliers, encode categorical features,
    perform feature engineering, and save the transformed data.
    """
    # Load data
    df = load_data(file_path)
    
    # Display basic info
    display_basic_info(df)
    
    # # Detect and visualize outliers
    # detect_outliers(df)
    # visualize_outliers(df)
    
    # Encode categorical features
    df_encoded = encode_categorical_features(df, categorical_features)
    
    # Perform feature engineering
    df_encoded = feature_engineering(df_encoded)
    
    # Save the transformed data (optional)
    if save_path:
        save_transformed_data(df_encoded, save_path)
    
    return df_encoded