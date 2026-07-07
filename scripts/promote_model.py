import os
import mlflow


def promote_model():
    # ------------------------------------------------------------
    # DagsHub / MLflow Configuration
    # ------------------------------------------------------------
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

    client = mlflow.MlflowClient()

    model_name = "model"

    # ------------------------------------------------------------
    # Get Latest Staging Model
    # ------------------------------------------------------------
    staging_versions = client.get_latest_versions(
        model_name,
        stages=["Staging"]
    )

    if not staging_versions:
        raise Exception(
            f"No model found in Staging for '{model_name}'."
        )

    latest_version_staging = staging_versions[0].version

    # ------------------------------------------------------------
    # Archive Existing Production Model
    # ------------------------------------------------------------
    production_versions = client.get_latest_versions(
        model_name,
        stages=["Production"]
    )

    for version in production_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )

    # ------------------------------------------------------------
    # Promote Staging Model to Production
    # ------------------------------------------------------------
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage="Production"
    )

    print(
        f"Model version {latest_version_staging} promoted to Production."
    )


if __name__ == "__main__":
    promote_model()