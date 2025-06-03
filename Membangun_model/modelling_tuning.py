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

# Set up DagsHub for MLflow tracking
dagshub.init(
    repo_owner="patuh", repo_name="SMSML_Patuh-Rujhan-Al-Istizhar", mlflow=True
)
mlflow.set_experiment("Diabetes Health Indicators")

# Load data
train_df = pd.read_csv("processed_data/train_processed.csv")
test_df = pd.read_csv("processed_data/test_processed.csv")

# Stratified sampling for train and test sets
train_df = train_df.groupby("Diabetes_012").sample(frac=0.5, random_state=42)
test_df = test_df.groupby("Diabetes_012").sample(frac=0.5, random_state=42)

X_train = train_df.drop(columns="Diabetes_012")
y_train = train_df["Diabetes_012"]
X_test = test_df.drop(columns="Diabetes_012")
y_test = test_df["Diabetes_012"]


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

    # Log each trial as a separate MLflow run
    with mlflow.start_run(run_name=f"LightGBM_Trial_{run_id}"):
        model = LGBMClassifier(**params)
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

        # Log params and metrics
        mlflow.log_params(params)
        mlflow.log_param("trial_id", run_id)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("train_time_seconds", train_time)

        # Save model
        joblib.dump(model, f"model_trial_{run_id}.joblib")
        mlflow.log_artifact(f"model_trial_{run_id}.joblib")

        # Save confusion matrix
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap="Blues")
        plt.tight_layout()
        plt.savefig(f"cm_trial_{run_id}.png")
        mlflow.log_artifact(f"cm_trial_{run_id}.png")
        plt.close(fig)

        return acc


# Run Optuna for 15 trials
study = optuna.create_study(direction="maximize")
start_tuning = time.time()
for i in range(15):
    study.optimize(lambda trial: objective(trial, i + 1), n_trials=1)
tuning_time = time.time() - start_tuning

# Train and log the best model
best_params = study.best_params
final_model = LGBMClassifier(**best_params, device="cpu", random_state=42)
start_time = time.time()
final_model.fit(X_train, y_train)
final_train_time = time.time() - start_time

# Evaluate the best model
preds = final_model.predict(X_test)
acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds, average="weighted", zero_division=0)
rec = recall_score(y_test, preds, average="weighted", zero_division=0)
f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

# Log the best model in MLflow
with mlflow.start_run(run_name="LightGBM_Final_Model"):
    mlflow.log_params(best_params)
    mlflow.log_param("tuning_method", "optuna")
    mlflow.log_param("num_trials", 15)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("train_time_seconds", final_train_time)
    mlflow.log_metric("tuning_time_seconds", tuning_time)

    # Save final model
    joblib.dump(final_model, "final_model.joblib")
    mlflow.log_artifact("final_model.joblib")

    # Save confusion matrix for final model
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=final_model.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues")
    plt.tight_layout()
    plt.savefig("cm_final.png")
    mlflow.log_artifact("cm_final.png")
    plt.close(fig)
