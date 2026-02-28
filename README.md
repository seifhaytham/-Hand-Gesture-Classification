# Hand Landmarks Classification with MLflow

This repository contains a Jupyter notebook and utility script for training and evaluating machine learning models on hand landmark data while tracking experiments using MLflow.

## Contents

- `ML_Project_MLflow.ipynb` - Main notebook demonstrating data preprocessing, model training (Random Forest, SVM, AdaBoost), hyperparameter tuning with `GridSearchCV`, and MLflow logging for metrics, parameters, artifacts, and model registration.
- `mlflow_utils.py` - Helper module that wraps common MLflow operations such as setting up experiments, starting runs, logging dataset information, model metrics/parameters, confusion matrix figures, comparison charts, and registering the best model.
- `hand_landmarks_data .csv` - Sample dataset containing 21 hand landmark coordinates and labels used for classification.
- `requirements.txt` - Python package dependencies required to run the notebook and script.

## Setup Instructions

1. **Create a Python virtual environment** (if not already done):
   ```powershell
   python -m venv env
   .\env\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Start MLflow tracking server** (default configuration uses a local file store under `mlruns`):
   ```powershell
   mlflow ui
   ```
   This will launch the MLflow UI at http://127.0.0.1:5000 by default. Ensure the tracking URI in the code points to the same address.

3. **Run the notebook** (`ML_Project_MLflow.ipynb`) cell-by-cell. The notebook:
   - Loads and preprocesses the dataset.
   - Normalizes coordinates relative to the wrist and mid-finger tip.
   - Splits data into train/test sets.
   - Executes model training using context managers that log to MLflow.
   - Compares model performances, logs charts and confusion matrices as artifacts.
   - Registers the best performing model in the MLflow Model Registry.

4. **Inspect results** using the MLflow UI. Each run should appear with logged parameters, metrics, and artifacts. The registered model will be available under the specified registry name.

## mlflow_utils.py Overview

Provides the following utilities:

- `setup_experiment(experiment_name)` – configure MLflow tracking URI and create/use experiment.
- `mlflow_run(run_name, tags)` – context manager for MLflow runs.
- `log_dataset_info()`, `log_model_run()`, `log_confusion_matrix_figure()`, `log_comparison_chart()` – logging helpers.
- `register_best_model()` – register a model build from a completed run.

## Important Notes

- Ensure the dataset path in the notebook (`DATASET_PATH`) is correct before running cells.
- The MLflow tracking URI is set in `mlflow_utils.setup_experiment`; modify if using a remote server.
- Deleting the `mlruns` directory will clear previous runs, and the notebook includes code to remove it at the start.

## Design Choices

- **Model selection:** Random Forest, SVM and AdaBoost were chosen to cover complementary algorithm families — tree-based ensembles (Random Forest), kernel methods (SVM) and boosting (AdaBoost). This gives a quick, pragmatic comparison between robust, high-bias/low-variance and high-variance/low-bias approaches.
- **Preprocessing:** Coordinates are re-centered and normalized relative to the wrist and mid-finger tip to remove translation and scale variations between samples, improving invariance to hand position and size.
- **Hyperparameter tuning:** `GridSearchCV` with a small cross-validation fold (`cv=3`) is used for reproducible, easy-to-understand tuning. The grid sizes are kept modest to balance compute time and coverage.
- **Metrics:** We log `test_accuracy` alongside macro and weighted F1/precision/recall to capture both overall correctness and per-class performance (useful when classes are imbalanced).
- **MLflow choices:** Local MLflow UI (`http://127.0.0.1:5000`) and a file-backed `mlruns` store are used for simple reproducibility and artifact inspection during development. Models and artifacts are logged per-run and the best model is registered to the Model Registry for later deployment.

## Contributing

Feel free to extend the notebook and utilities with additional models, evaluation metrics, or datasets. Create an issue or pull request for improvements.

---

This README provides a high-level guide for understanding and using the hand landmarks classification project with MLflow.
