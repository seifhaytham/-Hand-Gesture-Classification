## Hand Gesture Classification (Notebook)

This project is implemented **only in the notebook** `ML_Project.ipynb`. It trains and evaluates scikit‑learn models on a CSV of **MediaPipe hand landmarks** and includes a notebook section that runs **video inference** with MediaPipe Hands + OpenCV, overlaying the predicted gesture and confidence.

## What the notebook does

- **Dataset**: A CSV containing 21 hand landmarks \((x,y,z)\) per sample plus a `label`. The expected columns are:
  - `x1,y1,z1, x2,y2,z2, ... , x21,y21,z21, label`
- **Classes (18)**:
  - `call`, `dislike`, `fist`, `four`, `like`, `mute`, `ok`, `one`, `palm`,
    `peace`, `peace_inverted`, `rock`, `stop`, `stop_inverted`, `three`, `three2`, `two_up`, `two_up_inverted`
- **Preprocessing**: Re-centers landmarks at the wrist and normalizes coordinates relative to the middle-finger tip (per the notebook’s `preprocess_landmarks`).
- **Training / tuning**: Uses `GridSearchCV` to tune and compare:
  - Random Forest
  - SVM
  - AdaBoost
- **Evaluation**: Prints classification metrics and displays confusion matrices.
- **Model export**: Saves the best model with `joblib`.
- **Video demo**: The notebook class `GestureInference` loads the saved model, runs MediaPipe Hands on a video, and draws:
  - Hand landmarks using **MediaPipe drawing utilities**
  - Gesture label + confidence
  - Writes an annotated output video (default `gesture_demo.mp4`)

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to run

### Prerequisites

- Python 3.11 with Jupyter (or VS Code, Colab, etc.).
- Dependencies installed: `pip install -r requirements.txt`
- Your hand landmarks CSV file (columns: `x1,y1,z1` … `x21,y21,z21`, `label`).

---

### Part 1: Training the model

1. **Open the notebook**  
   Open `ML_Project.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.

2. **Set the data path**  
   In the cell that loads the CSV, update the path to your dataset:
   ```python
   df = pd.read_csv(r"C:\path\to\your\hand_landmarks_data.csv")
   ```
   The notebook expects columns `x1`–`x21`, `y1`–`y21`, `z1`–`z21`, and `label`.

3. **Run all cells up to and including training**  
   Execute cells in order:
   - Imports and constants
   - Load data and EDA
   - Preprocessing (feature extraction, train/test split)
   - Model training (Random Forest, SVM, AdaBoost with `GridSearchCV`)
   - Evaluation (classification reports, confusion matrices)
   - Model save: `joblib.dump(rf, 'random_forest_model.pkl')`

4. **Check the saved model**  
   After the save cell runs, `random_forest_model.pkl` should appear in the notebook’s working directory (usually the project folder).

---

### Part 2: Video inference demo

1. **Run the `GestureInference` class cell**  
   Execute the cell that defines the `GestureInference` class (it uses MediaPipe Hands and the drawing utilities).

2. **Configure paths in `main()`**  
   In the cell that defines and calls `main()`, set:
   - **`model_path`**: Path to the saved model. By default it uses `PROJECT_DIR / "random_forest_model.pkl"`, so if the model is in the same folder as the notebook, no change is needed.
   - **`video_file`**: Path to your input video, e.g.:
     ```python
     video_file = Path(r"C:\Users\You\Videos\my_hand_video.mp4")
     ```
   - **`output_file`**: Default is `PROJECT_DIR / "gesture_demo.mp4"`. Change this if you want a different output path.



4. **Run the `main()` cell**  
   Execute the cell that calls `main()`. The notebook will:
   - Load the model
   - Open the video
   - Run MediaPipe Hands on each frame
   - Draw hand landmarks and the predicted gesture with confidence
   - Write the annotated video to `gesture_demo.mp4` (or your chosen path)


---



## Files

- `ML_Project.ipynb`: training, evaluation, and video inference demo
- `requirements.txt`: dependencies used by the notebook
- `random_forest_model.pkl`: model saved by the notebook (created after training)



