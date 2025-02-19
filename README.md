# 5CCSAMLF_regression_model

1. Ensure all dependencies in the requirements.txt file have been installed.

2. Run file 'final_eval.py' to create a 'CW1_submission_K22015880.csv' with the predictions of outcome of the test set.

3. To see the process of how trainning data was processed and the suitable model was selected go to /experiments/ folder and look at the two notebooks (eda.ipynb and Model Training Pipeline.ipynb).

4. These two notebooks are cleaned up and have been split into 4 different python files in experiments folder. These files are:
    - data_preprocessing.py [feature engineering and data cleaning]
    - feature_selection.py [feature selection methods: eg RFECV, SFS, Correlation, Low Variance Filtering]
    - model_selection.py [comparison of different models such as Random Forest, Gradient Boosting, XGBoost, Linear Regression, CatBoost]
    - model_dev.py [Selected models are trained and go through hyperparameter tuning to find best model with best parameters]