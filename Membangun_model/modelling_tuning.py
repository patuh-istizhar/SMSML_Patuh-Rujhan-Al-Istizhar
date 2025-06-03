import time

import dagshub
import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV

# --- Konfigurasi MLflow Tracking dengan DagsHub ---
DAGSHUB_REPO_OWNER = "patuh"
DAGSHUB_REPO_NAME = "SMSML_Patuh-Rujhan-Al-Istizhar"

try:
    dagshub.init(
        repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True
    )
    print(
        f"MLflow tracking diinisialisasi dengan DagsHub: {DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}"
    )
except Exception as e:
    print(f"ERROR: Gagal menginisialisasi DagsHub MLflow tracking: {e}")
    print(
        "Pastikan DAGSHUB_REPO_OWNER dan DAGSHUB_REPO_NAME sudah benar di script ini."
    )
    print("Juga, pastikan kredensial DagsHub Anda sudah diatur.")
    exit()

# Mengatur nama eksperimen MLflow
mlflow.set_experiment("Diabetes Health Indicators")


# --- Fungsi Utama untuk Proses Tuning dan Logging ---
def main():
    # --- Muat Data ---
    try:
        train_data = pd.read_csv("processed_data/train_processed.csv")
        test_data = pd.read_csv("processed_data/test_processed.csv")
        print(
            f"Data train dimuat: {len(train_data)} baris. Data test dimuat: {len(test_data)} baris."
        )
    except FileNotFoundError as e:
        print(f"ERROR: File data tidak ditemukan: {e}")
        print(
            "Pastikan folder 'processed_data' berisi 'train_processed.csv' dan 'test_processed.csv'."
        )
        exit()

    TARGET_COLUMN = "Diabetes_012"

    X_train = train_data.drop(TARGET_COLUMN, axis=1)
    y_train = train_data[TARGET_COLUMN]

    X_test = test_data.drop(TARGET_COLUMN, axis=1)
    y_test = test_data[TARGET_COLUMN]

    # --- Definisi Ruang Parameter untuk LightGBM Classifier ---
    param_distributions = {
        "n_estimators": randint(100, 1000),
        "learning_rate": uniform(0.01, 0.2),
        "num_leaves": randint(20, 100),
        "max_depth": randint(5, 20),
        "min_child_samples": randint(20, 100),
        "subsample": uniform(0.6, 1.0),
        "colsample_bytree": uniform(0.6, 1.0),
        "reg_alpha": uniform(0.0, 0.5),
        "reg_lambda": uniform(0.0, 0.5),
        "device": ["gpu"],
        "max_bin": randint(200, 255),
    }

    # --- Inisialisasi RandomizedSearchCV ---
    print("\nMemulai Hyperparameter Tuning dengan RandomizedSearchCV...")
    tuned_model = RandomizedSearchCV(
        estimator=LGBMClassifier(random_state=42, device="gpu"),
        param_distributions=param_distributions,
        n_iter=50,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=3,
        random_state=42,
    )

    tuning_start_time = time.time()
    tuned_model.fit(X_train, y_train)
    tuning_total_time = time.time() - tuning_start_time

    print("\nTuning selesai. Parameter terbaik ditemukan:")
    print(tuned_model.best_params_)
    print(f"Skor Akurasi Terbaik (Cross-Validation): {tuned_model.best_score_:.4f}")
    print(f"Total waktu tuning: {tuning_total_time:.2f} detik")

    # --- Logging Model Terbaik ke MLflow ---
    with mlflow.start_run(run_name="Tuned LightGBM Model"):
        print("\nLogging model terbaik dan metrik ke MLflow (DagsHub)...")

        best_params = tuned_model.best_params_
        best_estimator = tuned_model.best_estimator_

        mlflow.log_params(best_params)
        mlflow.log_param("tuning_iterations", tuned_model.n_iter)
        mlflow.log_param("tuning_cv_folds", tuned_model.cv)

        # Latih Ulang Model Terbaik pada seluruh data training
        train_start_time = time.time()
        best_estimator.fit(X_train, y_train)
        training_time = time.time() - train_start_time

        # Prediksi dan Hitung Metrik pada Test Set
        y_pred = best_estimator.predict(X_test)

        # Waktu Inferensi
        inference_start_time = time.time()
        best_estimator.predict(X_test.head(100))
        inference_time_per_100_samples = time.time() - inference_start_time

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Ukuran Model
        model_size_bytes = len(joblib.dump(best_estimator))

        # Logging Metrik
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)
        mlflow.log_metric("training_time_sec", training_time)
        mlflow.log_metric("total_tuning_time_sec", tuning_total_time)
        mlflow.log_metric(
            "inference_time_per_100_samples_sec", inference_time_per_100_samples
        )
        mlflow.log_metric("model_size_bytes", model_size_bytes)

        print(f"Akurasi Model pada Test Set: {accuracy:.4f}")

        # Logging Model sebagai Artefak
        mlflow.sklearn.log_model(
            sk_model=best_estimator,
            artifact_path="lightgbm_model",
            input_example=X_train.head(5),
        )
        print("Model dilog sebagai artefak.")

        # Artefak Tambahan 1: Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=best_estimator.classes_
        )
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title("Confusion Matrix for Best LightGBM Model")
        cm_plot_path = "confusion_matrix_lightgbm.png"
        plt.savefig(cm_plot_path)
        mlflow.log_artifact(cm_plot_path)
        plt.close(fig)
        print("Confusion Matrix dilog sebagai artefak.")

        # Artefak Tambahan 2: Model dalam format Joblib
        model_filename = "best_lightgbm_model.joblib"
        joblib.dump(best_estimator, model_filename)
        mlflow.log_artifact(model_filename)
        print(f"Model juga dilog dalam format {model_filename}.")

        print("MLflow Run selesai.")


if __name__ == "__main__":
    main()
