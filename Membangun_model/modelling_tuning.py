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

# Konfigurasi DagsHub MLflow Tracking
dagshub.init(
    repo_owner="patuh", repo_name="SMSML_Patuh-Rujhan-Al-Istizhar", mlflow=True
)
mlflow.set_experiment("Diabetes Health Indicators")

# Muat dataset yang sudah diproses
train_df = pd.read_csv("processed_data/train_processed.csv")
test_df = pd.read_csv("processed_data/test_processed.csv")

X_train = train_df.drop(columns="Diabetes_012")
y_train = train_df["Diabetes_012"]

X_test = test_df.drop(columns="Diabetes_012")
y_test = test_df["Diabetes_012"]


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


# Jalankan tuning Optuna
study = optuna.create_study(direction="maximize")
start_tuning = time.time()
study.optimize(objective, n_trials=50)
tuning_duration = time.time() - start_tuning

# Ambil model terbaik
best_params = study.best_params
model = LGBMClassifier(**best_params, device="gpu", random_state=42)
start_train = time.time()
model.fit(X_train, y_train)
train_duration = time.time() - start_train

# Evaluasi
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds, average="weighted", zero_division=0)
rec = recall_score(y_test, preds, average="weighted", zero_division=0)
f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

# Logging ke MLflow
with mlflow.start_run(run_name="LightGBM with Optuna"):
    mlflow.log_params(best_params)
    mlflow.log_param("tuning_method", "optuna")
    mlflow.log_param("n_trials", 50)

    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric("test_precision", prec)
    mlflow.log_metric("test_recall", rec)
    mlflow.log_metric("test_f1_score", f1)
    mlflow.log_metric("training_time_sec", train_duration)
    mlflow.log_metric("tuning_time_sec", tuning_duration)

    # Simpan model
    joblib.dump(model, "best_model.joblib")
    mlflow.log_artifact("best_model.joblib")

    # Simpan confusion matrix
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close(fig)
