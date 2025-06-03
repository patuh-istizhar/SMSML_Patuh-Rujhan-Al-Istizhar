import os
import time

import dagshub
import joblib
import matplotlib.pyplot as plt
import mlflow
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# --- Setup DagsHub MLflow Tracking ---
DAGSHUB_REPO_OWNER = "patuh"
DAGSHUB_REPO_NAME = "SMSML_Patuh-Rujhan-Al-Istizhar"

try:
    dagshub.init(
        repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True
    )
    print(
        f"MLflow tracking initialized with DagsHub repo: {DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}"
    )
except Exception as e:
    print(f"Failed to initialize DagsHub MLflow tracking: {e}")
    exit()

mlflow.set_experiment("Diabetes Health Indicators with Optuna")


# --- Load and Prepare Data ---
def load_data():
    train_path = "processed_data/train_processed.csv"
    test_path = "processed_data/test_processed.csv"
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print(
            f"Data files not found. Please ensure '{train_path}' and '{test_path}' exist."
        )
        exit()

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    print(
        f"Loaded train data: {len(train_data)} rows, test data: {len(test_data)} rows."
    )

    TARGET = "Diabetes_012"
    X_train = train_data.drop(columns=[TARGET])
    y_train = train_data[TARGET]
    X_test = test_data.drop(columns=[TARGET])
    y_test = test_data[TARGET]

    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = load_data()


# --- Optuna Objective Function ---
def objective(trial):
    params = {
        "boosting_type": "gbdt",
        "device": "gpu",
        "gpu_use_dp": False,
        "tree_learner": "serial",
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "num_leaves": trial.suggest_int("num_leaves", 31, 150),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "max_bin": trial.suggest_int("max_bin", 200, 255),
        "random_state": 42,
        "n_jobs": -1,
    }

    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc


# --- Main Tuning and Logging ---
def main():
    study = optuna.create_study(direction="maximize")
    print("Starting Optuna hyperparameter tuning...")
    tuning_start = time.time()
    study.optimize(objective, n_trials=50)
    tuning_duration = time.time() - tuning_start

    best_params = study.best_trial.params
    print(f"Best params: {best_params}")
    print(f"Best accuracy (on test set): {study.best_value:.4f}")
    print(f"Total tuning time: {tuning_duration:.2f} seconds")

    # Add fixed params for GPU and reproducibility
    best_params.update(
        boosting_type="gbdt",
        device="gpu",
        gpu_use_dp=False,
        tree_learner="serial",
        random_state=42,
        n_jobs=-1,
    )

    model = LGBMClassifier(**best_params)

    with mlflow.start_run(run_name="LightGBM Optuna Tuning"):
        print("Training final model with best parameters...")
        train_start = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - train_start

        print("Evaluating model on test set...")
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Inference time for 100 samples
        inference_start = time.time()
        model.predict(X_test.head(100))
        inference_time_per_100 = time.time() - inference_start

        # Save model size
        model_path = "lightgbm_model.joblib"
        joblib.dump(model, model_path)
        model_size_bytes = os.path.getsize(model_path)

        # Logging parameters & metrics
        mlflow.log_params(best_params)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)
        mlflow.log_metric("training_time_sec", training_time)
        mlflow.log_metric("tuning_time_sec", tuning_duration)
        mlflow.log_metric("inference_time_per_100_samples_sec", inference_time_per_100)
        mlflow.log_metric("model_size_bytes", model_size_bytes)

        print(f"Test accuracy: {accuracy:.4f}")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train.head(5),
        )
        mlflow.log_artifact(model_path)

        # Confusion Matrix plot
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=model.classes_
        )
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title("Confusion Matrix - LightGBM Optuna")
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close(fig)
        mlflow.log_artifact(cm_path)

        print("Run complete, artifacts and metrics logged to MLflow.")


if __name__ == "__main__":
    main()
