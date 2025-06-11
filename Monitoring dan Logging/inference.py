import requests

url = "http://127.0.0.1:8000/predict"

payload = {
    "dataframe_split": {
        "columns": [
            "x0",
            "x1",
            "x2",
            "x3",
            "x4",
            "x5",
            "x6",
            "x7",
            "x8",
            "x9",
            "x10",
            "x11",
            "x12",
            "x13",
            "x14",
            "x15",
            "x16",
            "x17",
            "x18",
            "x19",
            "x20",
        ],
        "data": [
            [
                -0.2857142857142857,
                0,
                0,
                0,
                0.5,
                -0.5,
                -1,
                1,
                0,
                1,
                0,
                0,
                0,
                1,
                1,
                1,
                0,
                1,
                0,
                0,
                1,
            ]
        ],
    }
}

headers = {"Content-Type": "application/json"}

try:
    response = requests.post(url, json=payload, headers=headers)

    print("Status code:", response.status_code)
    print("Response body:", response.text)

except requests.exceptions.ConnectionError as e:
    print(f"Error connecting to exporter/model server: {e}")
    print(
        "Please ensure your prometheus_exporter.py script is running at http://127.0.0.1:8000"
    )
    print("And your MLflow model serving is running (e.g., at http://127.0.0.1:5005).")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
