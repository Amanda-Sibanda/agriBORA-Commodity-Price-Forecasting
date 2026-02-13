# ==========================
# FULL TRAINING PIPELINE WITH MLFLOW
# ==========================

# --------------------------
# 1. Imports
# --------------------------
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
import mlflow
import mlflow.sklearn
import os

# --------------------------
# 2. Select Model
# --------------------------
model_choice = 'cb'  # Options: 'rf', 'gb', 'cb', 'lgbm'

# --------------------------
# 3. Load Data
# --------------------------
path = "C:/Users/Dell/Downloads/training/"

X_train = pd.read_csv(path + 'X_train_20260212.csv')
y_train = pd.read_csv(path + 'y_train_20260212.csv')
X_test = pd.read_csv(path + 'X_test_20260212.csv')
y_test = pd.read_csv(path + 'y_test_20260212.csv')
X_eval = pd.read_csv(path + 'X_eval_20260212.csv')
y_eval = pd.read_csv(path + 'y_eval_20260212.csv')

y_train = y_train.iloc[:, 0]
y_test = y_test.iloc[:, 0]
y_eval = y_eval.iloc[:, 0]

# --------------------------
# 4. Define models and grid
# --------------------------
models = {
    'rf': RandomForestRegressor(random_state=42, n_jobs=-1),
    'gb': GradientBoostingRegressor(random_state=42),
    'cb': CatBoostRegressor(random_state=42, silent=True),
    'lgbm': lgb.LGBMRegressor(random_state=42, n_jobs=-1)
}

grids = {
    'rf': {
        'n_estimators': [400, 800],
        'max_depth': [None, 15, 25],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 3, 5],
        'max_features': ['sqrt', 0.8]
    },
    'gb': {
        'n_estimators': [400, 800],
        'learning_rate': [0.01, 0.03, 0.05],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'min_samples_leaf': [1, 3, 5]
    },
    'cb': {
        'iterations': [600, 1000],
        'learning_rate': [0.01, 0.03, 0.05],
        'depth': [4, 6, 8],
        'l2_leaf_reg': [1, 3, 5, 10],
        'bagging_temperature': [0, 1]
    },
    'lgbm': {
        'n_estimators': [600, 1000],
        'learning_rate': [0.01, 0.03, 0.05],
        'max_depth': [-1, 15, 25],
        'num_leaves': [31, 63, 127],
        'min_child_samples': [10, 20, 50],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [0, 0.1, 1]
    }
}

# --------------------------
# 5. MLflow Setup
# --------------------------
mlflow.set_tracking_uri("file:///C:/Users/Dell/Downloads/training/mlruns")
mlflow.set_experiment("Wholesale_Agribora_Price_Forecast")

# --------------------------
# 6. Sample Hyperparameter Combinations
# --------------------------
n_combinations = 144  # <-- Set the number of combinations manually
param_grid = grids[model_choice]

param_list = list(ParameterSampler(
    param_grid,
    n_iter=n_combinations,
    random_state=42
))

# --------------------------
# 7. Train and Track with MLflow
# --------------------------
for i, params in enumerate(param_list):

    with mlflow.start_run(run_name=f"{model_choice}_run_{i}"):

        # Refit model with this parameter combination
        if model_choice == 'cb':
            model = CatBoostRegressor(random_state=42, silent=True, **params)
        else:
            model = models[model_choice].set_params(**params)
        model.fit(X_train, y_train)
        
        # Log parameters
        mlflow.log_params(params)

        # --------------------------
        # Feature Importance (Graph - ALL features)
        # --------------------------
        if hasattr(model, "feature_importances_"):

            fi = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)

            plt.figure(figsize=(10, max(6, len(fi) * 0.25)))
            plt.barh(fi['feature'], fi['importance'])
            plt.title(f"Feature Importances ({model_choice})")
            plt.xlabel("Importance")
            plt.tight_layout()

            fi_plot_path = f"feature_importance_run_{i}.png"
            plt.savefig(fi_plot_path)
            plt.close()
            mlflow.log_artifact(fi_plot_path)

        # --------------------------
        # Residuals & Pred vs Actual (always log)
        # --------------------------
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred

        # Residual histogram
        plt.figure(figsize=(10,5))
        sns.histplot(residuals, kde=True)
        plt.title("Residuals Distribution")
        residual_plot_path = f"residuals_run_{i}.png"
        plt.savefig(residual_plot_path)
        plt.close()
        mlflow.log_artifact(residual_plot_path)

        # Predicted vs Actual
        plt.figure(figsize=(12,5))
        plt.plot(y_test.values, label="Actual")
        plt.plot(y_pred, label="Predicted")
        plt.legend()
        plt.title("Predicted vs Actual")
        pred_plot_path = f"pred_vs_actual_run_{i}.png"
        plt.savefig(pred_plot_path)
        plt.close()
        mlflow.log_artifact(pred_plot_path)

        # --------------------------
        # Log evaluation metrics
        # --------------------------
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("Test_MAE", mae)
        mlflow.log_metric("Test_RMSE", rmse)
        mlflow.log_metric("Test_R2", r2)

        print(f"Logged run {i} | Params: {params}")

print("Training complete. Open MLflow UI on port 5000 to explore runs and metrics.")
