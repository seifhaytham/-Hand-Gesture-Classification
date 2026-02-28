# Hand Gesture Classification

This repository implements a complete pipeline for training and evaluating hand‑gesture classifiers, then applying a trained model to annotate video files. It uses **MediaPipe** to extract 21 hand landmarks per frame, and **scikit‑learn** classifiers (Random Forest, SVM, AdaBoost) for prediction.

## Overview

The notebook loads a CSV dataset (`hand_landmarks_data .csv`), performs EDA and preprocessing (centering/normalizing landmarks), and fits multiple models using `GridSearchCV`. After training, it prints classification reports, plots confusion matrices, and shows precision‑recall curves for each model. The best model is saved (e.g. `random_forest_model.pkl`).

The `gesture_video_inference.py` script loads a saved model and processes a video file frame‑by‑frame, drawing the detected hand skeleton and predicted gesture with a confidence score. Results are written to a new output video.

## Features

- **Video File Inference**: Annotate recorded videos with gesture predictions.
- **18 Gesture Classes**: 
  `call`, `dislike`, `fist`, `four`, `like`, `mute`, `ok`, `one`, `palm`,
  `peace`, `peace_inverted`, `rock`, `stop`, `stop_inverted`, `three`, `three2`, `two_up`, `two_up_inverted`.
- **Robust Feature Extraction**: Relative landmark coordinates normalized against wrist and middle finger tip.
- **Model Comparison**: Train and compare Random Forest, SVM, and AdaBoost classifiers.
- **Evaluation Visuals**: Notebook includes classification reports, confusion matrices, and precision‑recall plots for each model.
- **Inference Smoothing**: Video script smooths predictions over a short history to reduce flicker.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/hand-gesture-classification.git
   cd hand-gesture-classification
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Training the Model

1. Open `ML_Project.ipynb` in Jupyter.
2. Execute cells sequentially to load data, preprocess, and train models.
3. Review outputs:
   - Classification reports for each classifier.
   - Confusion matrices plotted with `ConfusionMatrixDisplay`.
   - Precision‑recall curves generated with `PrecisionRecallDisplay`.
4. The notebook saves the selected `random_forest_model.pkl` (or whichever estimator you choose).

### 2. Video Inference

Edit the `main()` section of `gesture_video_inference.py` to match your file locations:

```python
PROJECT_DIR = Path(__file__).parent
model_path = Path(r"C:\Users\Dell\Desktop\ml-project\random_forest_model.pkl")
video_file = Path(r"C:\Users\Dell\Pictures\Camera Roll\WIN_20260225_12_13_12_Pro.mp4")

inference = GestureInference(str(model_path), smoothing_window=3)
result_path = inference.process_video(
    str(video_file),
    str(PROJECT_DIR / "gesture_demo.mp4"),
    skip_frames=2  # process every 2nd frame for speed
)
```

Parameters you can tweak:
- `smoothing_window`: number of recent frames used to average the prediction.
- `skip_frames`: process every *n*‑th frame to speed up inference.

Run the script:
```bash
python gesture_video_inference.py
```
An output video (`gesture_demo.mp4` by default) will be created along with console logs showing progress.

## Project Structure

- `ML_Project.ipynb` – notebook covering data loading, training, and evaluation.
- `gesture_video_inference.py` – inference script for video processing.
- `requirements.txt` – Python dependencies.
- `hand_landmarks_data .csv` – dataset of landmarks and labels.
- `random_forest_model.pkl` – placeholder for the saved classifier.
- `gesture_demo.mp4` – example output video with annotated gestures.


