from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
import time
import dagshub
import re
import string
import numpy as np

from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import warnings
warnings.filterwarnings("ignore")


# -------------------------
# TEXT PREPROCESSING
# -------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    text = " ".join([w for w in text.split() if w not in stop_words])
    text = " ".join([lemmatizer.lemmatize(w) for w in text.split()])
    return text


# -------------------------
# MLflow Setup
# -------------------------
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
repo_owner = "abhishekla311"
repo_name = "capston_project"

mlflow.set_tracking_uri(
    f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
)



# -------------------------
# LOAD MODEL
# -------------------------
MODEL_NAME = "model"

client = mlflow.MlflowClient()
versions = client.search_model_versions(f"name='{MODEL_NAME}'")

if not versions:
    raise ValueError(f"Model '{MODEL_NAME}' not found in registry")

latest = max(versions, key=lambda x: int(x.version))
model_uri = f"models:/{MODEL_NAME}/{latest.version}"

print("Loading model:", model_uri)

model = mlflow.pyfunc.load_model(model_uri)

vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))


# -------------------------
# FLASK APP
# -------------------------
app = Flask(__name__)

registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "app_request_count",
    "Requests",
    ["method", "endpoint"],
    registry=registry
)

REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Latency",
    ["endpoint"],
    registry=registry
)

PREDICTION_COUNT = Counter(
    "prediction_count",
    "Predictions",
    ["label"],
    registry=registry
)


# -------------------------
# ROUTES
# -------------------------
@app.route("/")
def home():
    return render_template("index.html", result=None)


@app.route("/predict", methods=["POST"])
def predict():

    start = time.time()

    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()

    text = request.form.get("text", "")
    text = normalize_text(text)

    features = vectorizer.transform([text])
    df = pd.DataFrame(features.toarray())

    prediction = model.predict(df)[0]

    PREDICTION_COUNT.labels(label=str(prediction)).inc()

    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start)

    return render_template("index.html", result=prediction)


@app.route("/metrics")
def metrics():
    return generate_latest(registry), 200, {
        "Content-Type": CONTENT_TYPE_LATEST
    }


# -------------------------
# RUN APP
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)