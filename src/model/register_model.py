import os
import sys
import json
import mlflow
import dagshub
import warnings

# Ignore warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.logger import logging

# -------------------------
# DagsHub Credentials
# -------------------------
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
repo_owner = "abhishekla311"
repo_name = "capston_project"




dagshub_url = "https://dagshub.com/abhishekla311/capston_project.mlflow"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
dagshub.init(
    repo_owner=repo_owner,
    repo_name=repo_name,
    mlflow=True
)

# -------------------------
# Load model info
# -------------------------
def load_model_info(file_path: str) -> dict:
    try:
        with open(file_path, "r") as file:
            data = json.load(file)

        logging.info("Model info loaded from %s", file_path)
        return data

    except Exception as e:
        logging.exception(e)
        raise


# -------------------------
# Register model
# -------------------------
def register_model(model_name: str, model_info: dict):

    try:
        # MLflow 2.22.0 correct URI format
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        logging.info("Registering model from URI: %s", model_uri)

        result = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        logging.info(
            "Model registered successfully. Version: %s",
            result.version
        )

        # Optional: transition to Staging (works in MLflow 2.x)
        client = mlflow.tracking.MlflowClient()

        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage="Staging"
        )

        logging.info("Model moved to Staging stage.")

    except Exception as e:
        logging.exception("Model registration failed: %s", e)
        raise


# -------------------------
# Main function
# -------------------------
def main():

    try:
        model_info = load_model_info(
            "reports/experiment_info.json"
        )

        register_model(
            model_name="model",
            model_info=model_info
        )

    except Exception as e:
        logging.exception("Pipeline failed: %s", e)
        print("Error:", e)


if __name__ == "__main__":
    main()