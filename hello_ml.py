import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

def run_pipeline():
    print("--- Starting ML Pipeline ---")

    # 1. Check for Hardware Acceleration (CUDA)
    # This is helpful for your Docker/VMware testing
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device = "cuda" if cuda_available else "cpu"
        print(f"Device detected: {device}")
    except ImportError:
        device = "cpu"
        print("Torch not installed, defaulting to CPU.")

    # 2. Generate Synthetic Data (using your Numpy 2.4/Pandas 1.5 setup)
    print("Generating synthetic dataset...")
    X, y = np.random.standard_normal(size=(1000, 10)), np.random.standard_normal(size=(1000,))
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

    # 4. Train XGBoost Model
    # We use 'hist' for CPU and 'gpu_hist' if you eventually get CUDA working
    print(f"Training XGBoost model on {device}...")
    model = xgb.XGBRegressor(
        tree_method="hist" if device == "cpu" else "gpu_hist",
        n_estimators=100,
        learning_rate=0.1
    )
    model.fit(X_train, y_train)

    # 5. Evaluate
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"Model Training Complete. RMSE: {rmse:.4f}")

    # 6. Save Model (using Joblib from your requirements)
    os.makedirs("models", exist_ok=True)
    model_path = "models/xgboost_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    print("--- Pipeline Finished Successfully ---")

if __name__ == "__main__":
    run_pipeline()