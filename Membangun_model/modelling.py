import mlflow
import mlflow.sklearn
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

# --- Konfigurasi MLflow Tracking Lokal ---
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Diabetes Health Indicators")

# --- Muat Data ---
try:
    train_data = pd.read_csv("processed_data/train_processed.csv")
    test_data = pd.read_csv("processed_data/test_processed.csv")
    print("Data train dan test berhasil dimuat.")
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

# --- MLflow Run ---
with mlflow.start_run():
    print("Memulai MLflow Run untuk model dasar...")

    # Mengaktifkan autologging untuk Scikit-learn
    # Ini akan secara otomatis mencatat parameter, metrik, dan model
    mlflow.sklearn.autolog(log_input_examples=True)

    # --- Inisialisasi dan Pelatihan Model dengan Parameter ---
    params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "random_state": 42,
        "n_jobs": -1,
        "device": "gpu",
    }

    print(f"Melatih LGBMClassifier dengan parameter: {params}")
    model = LGBMClassifier(**params)

    model.fit(X_train, y_train)
    print("Model selesai dilatih.")

    # --- Evaluasi Model ---
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Akurasi Model pada Test Set: {accuracy:.4f}")

    print("MLflow Run selesai.")
