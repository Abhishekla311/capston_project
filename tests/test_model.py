# Load Test + Signature Test + Performance Test

import os
import pickle
import unittest

import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # -------------------------------
        # DagsHub / MLflow Configuration
        # -------------------------------
        repo_owner = "abhishekla311"
        repo_name = "capston_project"

        dagshub_token = os.getenv("CAPSTONE_TEST")

        if not dagshub_token:
            raise EnvironmentError(
                "CAPSTONE_TEST environment variable is not set."
            )

        os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        mlflow.set_tracking_uri(
            f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
        )

        # -------------------------------
        # Load Model
        # -------------------------------
        cls.new_model_name = "model"

        cls.new_model_version = cls.get_latest_model_version(
            cls.new_model_name
        )

        if cls.new_model_version is None:
            raise Exception(
                f"No version found for model '{cls.new_model_name}'."
            )

        cls.new_model_uri = (
            f"models:/{cls.new_model_name}/{cls.new_model_version}"
        )

        cls.new_model = mlflow.pyfunc.load_model(
            cls.new_model_uri
        )

        # -------------------------------
        # Load Vectorizer
        # -------------------------------
        with open("models/vectorizer.pkl", "rb") as f:
            cls.vectorizer = pickle.load(f)

        # -------------------------------
        # Load Holdout Dataset
        # -------------------------------
        cls.holdout_data = pd.read_csv(
            "data/processed/test_bow.csv"
        )

    @staticmethod
    def get_latest_model_version(
        model_name,
        stage="Staging",
    ):
        client = mlflow.MlflowClient()

        latest_versions = client.get_latest_versions(
            model_name,
            stages=[stage],
        )

        if not latest_versions:
            return None

        return latest_versions[0].version

    # ---------------------------------
    # Test 1 : Model Loading
    # ---------------------------------
    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    # ---------------------------------
    # Test 2 : Model Signature
    # ---------------------------------
    def test_model_signature(self):

        input_text = "hi how are you"

        input_data = self.vectorizer.transform(
            [input_text]
        )

        input_df = pd.DataFrame(
            input_data.toarray(),
            columns=[
                str(i)
                for i in range(input_data.shape[1])
            ],
        )

        prediction = self.new_model.predict(
            input_df
        )

        self.assertEqual(
            input_df.shape[1],
            len(
                self.vectorizer.get_feature_names_out()
            ),
        )

        self.assertEqual(
            len(prediction),
            input_df.shape[0],
        )

        self.assertEqual(
            len(prediction.shape),
            1,
        )

    # ---------------------------------
    # Test 3 : Performance
    # ---------------------------------
    def test_model_performance(self):

        X_holdout = self.holdout_data.iloc[:, :-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        y_pred = self.new_model.predict(
            X_holdout
        )

        accuracy = accuracy_score(
            y_holdout,
            y_pred,
        )

        precision = precision_score(
            y_holdout,
            y_pred,
        )

        recall = recall_score(
            y_holdout,
            y_pred,
        )

        f1 = f1_score(
            y_holdout,
            y_pred,
        )

        self.assertGreaterEqual(
            accuracy,
            0.40,
            "Accuracy is below threshold.",
        )

        self.assertGreaterEqual(
            precision,
            0.40,
            "Precision is below threshold.",
        )

        self.assertGreaterEqual(
            recall,
            0.40,
            "Recall is below threshold.",
        )

        self.assertGreaterEqual(
            f1,
            0.40,
            "F1 Score is below threshold.",
        )


if __name__ == "__main__":
    unittest.main()