

import os
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from contextlib import contextmanager


# ── Experiment Setup ──────────────────────────────────────────────────────────

def setup_experiment(experiment_name: str = "Hand-Landmarks-Classification") -> str:

    mlflow.set_tracking_uri("http://127.0.0.1:5000")          

    uri = mlflow.get_tracking_uri()
    if uri.startswith("file://") or uri == "mlruns":
        path = uri.replace("file://", "")
        os.makedirs(path, exist_ok=True)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"[MLflow] Created experiment '{experiment_name}' (id={experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"[MLflow] Using existing experiment '{experiment_name}' (id={experiment_id})")
    mlflow.set_experiment(experiment_name)
    return experiment_id



@contextmanager
def mlflow_run(run_name: str, tags: dict = None):
    """
    Context manager that wraps a training block in an MLflow run.

    Usage:
        with mlflow_run("Random-Forest-GridSearch") as run:
            log_dataset_info(X_train, X_test, dataset_path)
            ... train model ...
            log_model_run(model, params, y_test, y_pred, model_name="random_forest")
    """
    with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
        print(f"[MLflow] Run started: '{run_name}' (run_id={run.info.run_id})")
        yield run
        print(f"[MLflow] Run finished: '{run_name}'")



def log_dataset_info(X_train, X_test, dataset_path: str = None):
    """Log dataset shape and optional file path as MLflow params/tags."""
    mlflow.log_param("dataset_train_samples", X_train.shape[0])
    mlflow.log_param("dataset_test_samples", X_test.shape[0])
    mlflow.log_param("dataset_features", X_train.shape[1])
    if dataset_path:
        mlflow.log_param("dataset_path", dataset_path)
        # Log the CSV itself as an artifact so the run is self-contained
        if os.path.exists(dataset_path):
            mlflow.log_artifact(dataset_path, artifact_path="dataset")


def log_model_run(model, best_params: dict, y_test, y_pred,
                  model_name: str, cv_score: float = None):
 
    # -- Parameters
    for key, value in best_params.items():
        mlflow.log_param(key, value)

    # -- Metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("macro_f1", report["macro avg"]["f1-score"])
    mlflow.log_metric("macro_precision", report["macro avg"]["precision"])
    mlflow.log_metric("macro_recall", report["macro avg"]["recall"])
    mlflow.log_metric("weighted_f1", report["weighted avg"]["f1-score"])
    if cv_score is not None:
        mlflow.log_metric("best_cv_score", cv_score)

    print(f"  test_accuracy  = {accuracy:.4f}")
    print(f"  macro_f1       = {report['macro avg']['f1-score']:.4f}")

    # -- Model artifact
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=model_name,
        registered_model_name=None,     # registration done separately via register_best_model()
    )


def log_confusion_matrix_figure(fig, model_name: str):

    artifact_filename = f"confusion_matrix_{model_name}.png"
    fig.savefig(artifact_filename, bbox_inches="tight")
    mlflow.log_artifact(artifact_filename, artifact_path="confusion_matrices")
    os.remove(artifact_filename)
    print(f"  [MLflow] Confusion matrix logged for '{model_name}'")


def log_comparison_chart(results: dict):
    """
    Create and log a bar-chart comparing test accuracy across all models.

    Parameters
    ----------
    results : dict  e.g. {"Random Forest": 0.97, "SVM": 0.96, "AdaBoost": 0.94}
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    models = list(results.keys())
    scores = list(results.values())
    bars = ax.bar(models, scores, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Model Comparison – Test Accuracy")
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{score:.4f}", ha="center", va="bottom", fontsize=11)
    plt.tight_layout()
    plt.savefig("model_comparison.png", bbox_inches="tight")
    plt.show()

    # Log the chart as a standalone artifact (outside any run, or call inside a run)
    with mlflow.start_run(run_name="Model-Comparison-Chart", nested=True):
        mlflow.log_artifact("model_comparison.png", artifact_path="charts")
    os.remove("model_comparison.png")
    print("[MLflow] Comparison chart logged.")


# ── Model Registry ────────────────────────────────────────────────────────────

def register_best_model(run_id: str, model_artifact_path: str,
                        registry_name: str = "Hand-Landmarks-Best-Model"):
    """
    Register a model from a completed run into the MLflow Model Registry.

    Parameters
    ----------
    run_id               : run_id string from the winning run
    model_artifact_path  : artifact path used in log_model (e.g. 'random_forest')
    registry_name        : display name in the registry
    """
    model_uri = f"runs:/{run_id}/{model_artifact_path}"
    result = mlflow.register_model(model_uri=model_uri, name=registry_name)
    print(f"[MLflow] Registered model '{registry_name}' version {result.version}")

    client = MlflowClient()
    client.update_registered_model(
        name=registry_name,
        description="Best performing model selected from Random Forest, SVM, and AdaBoost comparison."
    )
    return result
