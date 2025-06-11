import json
import time

import dagshub
import matplotlib.pyplot as plt
import mlflow
import numpy as np
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

# Set up DagsHub for MLflow tracking
dagshub.init(
    repo_owner="patuh", repo_name="SMSML_Patuh-Rujhan-Al-Istizhar", mlflow=True
)
mlflow.set_experiment("Diabetes Health Indicators")

# Load data
train_df = pd.read_csv("diabetes_health_indicators_preprocessing/train_processed.csv")
test_df = pd.read_csv("diabetes_health_indicators_preprocessing/test_processed.csv")

X_train = train_df.drop(columns="Diabetes_012")
y_train = train_df["Diabetes_012"]
X_test = test_df.drop(columns="Diabetes_012")
y_test = test_df["Diabetes_012"]

# Prepare an input example for MLflow model signature
input_example = X_train.head(1)


def objective(trial, run_id):
    params = {
        "boosting_type": "gbdt",
        "device": "cpu",
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.15),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 80),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 0.5),
        "random_state": 42,
        "n_jobs": -1,
    }

    with mlflow.start_run(run_name=f"LightGBM_Trial_{run_id}"):
        model = LGBMClassifier(**params)
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Measure prediction time
        start_predict = time.time()
        preds = model.predict(X_test)
        predict_time = time.time() - start_predict

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

        # Log parameters and metrics
        mlflow.log_params(params)
        mlflow.log_param("trial_id", run_id)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("train_time_seconds", train_time)
        mlflow.log_metric("predict_time_seconds", predict_time)

        # Log metrics as JSON
        metrics_dict = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "train_time_seconds": train_time,
            "predict_time_seconds": predict_time,
        }
        mlflow.log_text(json.dumps(metrics_dict, indent=2), "metric_info.json")

        # Log model with input example
        mlflow.sklearn.log_model(
            model, f"model_trial_{run_id}", input_example=input_example
        )

        # Log confusion matrix
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap="Blues")
        plt.tight_layout()
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        # Log feature importance plot
        fig, ax = plt.subplots(figsize=(10, 6))
        importances = model.feature_importances_
        feature_names = X_train.columns
        indices = np.argsort(importances)[::-1]
        ax.bar(range(len(importances)), importances[indices], align="center")
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels(feature_names[indices], rotation=90)
        ax.set_title(f"Feature Importance - Trial {run_id}")
        plt.tight_layout()
        mlflow.log_figure(fig, "feature_importance.png")
        plt.close(fig)

        return acc


# Run Optuna
study = optuna.create_study(direction="maximize")
for i in range(15):
    study.optimize(lambda trial: objective(trial, i + 1), n_trials=1)
