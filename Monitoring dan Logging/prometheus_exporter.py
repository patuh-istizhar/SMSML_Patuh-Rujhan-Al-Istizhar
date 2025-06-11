import json
import time

import psutil
import requests
from flask import Flask, Response, jsonify, request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

app = Flask(__name__)

# Metrik untuk API model
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP Requests")
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP Request Latency")
THROUGHPUT = Counter("http_requests_throughput", "Total number of requests per second")
ML_MODEL_PREDICTION_SUCCESS = Counter(
    "ml_model_prediction_success_total", "Total successful ML model predictions"
)
ML_MODEL_PREDICTION_FAILURE = Counter(
    "ml_model_prediction_failure_total", "Total failed ML model predictions"
)

# Metrik untuk sistem
CPU_USAGE = Gauge("system_cpu_usage_percent", "CPU Usage Percentage")
RAM_USAGE = Gauge("system_ram_usage_percent", "RAM Usage Percentage")
DISK_USAGE = Gauge("system_disk_usage_percent", "Disk Usage Percentage")

# Metrik tambahan
NETWORK_BYTES_SENT = Gauge(
    "system_network_bytes_sent_total", "Total network bytes sent"
)
NETWORK_BYTES_RECV = Gauge(
    "system_network_bytes_recv_total", "Total network bytes received"
)
ML_MODEL_INPUT_PAYLOAD_SIZE_BYTES = Histogram(
    "ml_model_input_payload_size_bytes",
    "Size of input payload in bytes",
    buckets=(100, 500, 1000, 5000, 10000, float("inf")),
)

# Metrik untuk mengukur uptime aplikasi
APP_UPTIME_SECONDS = Gauge(
    "app_uptime_seconds", "Uptime of the Prometheus Exporter application in seconds"
)
START_TIME = time.time()


# Endpoint untuk Prometheus
@app.route("/metrics", methods=["GET"])
def metrics():
    # Update metrik sistem setiap kali /metrics diakses
    CPU_USAGE.set(psutil.cpu_percent(interval=1))
    RAM_USAGE.set(psutil.virtual_memory().percent)
    DISK_USAGE.set(psutil.disk_usage("/").percent)

    net_io = psutil.net_io_counters()
    # Mengatur nilai Gauge langsung dari psutil
    NETWORK_BYTES_SENT.set(net_io.bytes_sent)
    NETWORK_BYTES_RECV.set(net_io.bytes_recv)

    # Update uptime
    APP_UPTIME_SECONDS.set(time.time() - START_TIME)

    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


# Endpoint untuk mengakses API model dan mencatat metrik
@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()
    THROUGHPUT.inc()

    api_url = "http://127.0.0.1:5005/invocations"
    data = request.get_json()

    # Catat ukuran payload input
    if data:
        ML_MODEL_INPUT_PAYLOAD_SIZE_BYTES.observe(len(json.dumps(data).encode("utf-8")))

    try:
        response = requests.post(api_url, json=data)
        duration = time.time() - start_time
        REQUEST_LATENCY.observe(duration)

        if response.status_code == 200:
            ML_MODEL_PREDICTION_SUCCESS.inc()
        else:
            ML_MODEL_PREDICTION_FAILURE.inc()

        return jsonify(response.json())

    except Exception as e:
        ML_MODEL_PREDICTION_FAILURE.inc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
