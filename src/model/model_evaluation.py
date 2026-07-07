import os
import sys
import json
import pickle
from dotenv import load_dotenv


import dagshub
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Import logger
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
)
from src.logger import logging
load_dotenv()
# ---------------------------------------------------------------------
# DagsHub Credentials
# ---------------------------------------------------------------------
# Prefer reading these from your environment rather than hardcoding them.

dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token



# os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv(
#     "MLFLOW_TRACKING_USERNAME", ""sfef
# )
# os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv(
#     "MLFLOW_TRACKING_PASSWORD", ""
# )

# ---------------------------------------------------------------------
# MLflow Configuration
# ---------------------------------------------------------------------
repo_owner = "abhishekla311"
repo_name = "capston_project"

mlflow.set_tracking_uri(
    f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
)

dagshub.init(
    repo_owner=repo_owner,
    repo_name=repo_name,
    mlflow=True,
)

# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------
def load_model(file_path: str):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def load_data(file_path: str):
    return pd.read_csv(file_path)


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray):

    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred_prob),
    }

    return metrics


def save_output(data, file_path, is_json=True):

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if is_json:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
    else:
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    logging.info("Saved file : %s", file_path)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():

    mlflow.set_experiment("my-dvc-pipeline")

    with mlflow.start_run() as run:

        try:

            logging.info("Loading trained model...")

            clf = load_model("./models/model.pkl")

            logging.info("Loading test dataset...")

            test_df = load_data("./data/processed/test_bow.csv")

            X_test = test_df.iloc[:, :-1].values
            y_test = test_df.iloc[:, -1].values

            logging.info("Evaluating model...")

            metrics = evaluate_model(clf, X_test, y_test)

            save_output(
                metrics,
                "reports/metrics.json"
            )

            # ---------------------------------------------------------
            # Log Metrics
            # ---------------------------------------------------------
            mlflow.log_metrics(metrics)

            # ---------------------------------------------------------
            # Log Parameters
            # ---------------------------------------------------------
            if hasattr(clf, "get_params"):
                mlflow.log_params(clf.get_params())

            # ---------------------------------------------------------
            # Log Model (MLflow 2.22.0)
            # ---------------------------------------------------------
            mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path="model",
            )

            logging.info("Model logged successfully.")

            # ---------------------------------------------------------
            # Save experiment information
            # ---------------------------------------------------------
            experiment_info = {
                "run_id": run.info.run_id,
                "model_path": "model",
            }

            save_output(
                experiment_info,
                "reports/experiment_info.json",
            )

            # ---------------------------------------------------------
            # Log artifacts
            # ---------------------------------------------------------
            mlflow.log_artifact(
                "reports/metrics.json"
            )

            logging.info("Model evaluation completed successfully.")

        except Exception as e:
            logging.exception(e)
            raise


if __name__ == "__main__":
    main()