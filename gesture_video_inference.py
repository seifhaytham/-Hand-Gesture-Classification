"""
Hand Gesture Classification Video Inference
Demonstrates trained ML model on video with hand landmark detection.
Uses MediaPipe for landmark extraction and displays predictions.
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
from pathlib import Path
from collections import deque
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Gesture classes
GESTURE_CLASSES = [
    'call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one', 'palm',
    'peace', 'peace_inverted', 'rock', 'stop', 'stop_inverted', 'three',
    'three2', 'two_up', 'two_up_inverted'
]

# Hand skeleton connections for visualization
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


class GestureInference:
    """Hand gesture inference pipeline."""
    
    def __init__(self, model_path: str, smoothing_window: int = 3):
        """Initialize inference engine."""
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = joblib.load(str(self.model_path))
        print(f"[OK] Model loaded: {self.model_path.name}")
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        
        # Smoothing
        self.pred_history = deque(maxlen=smoothing_window)
        self.conf_history = deque(maxlen=smoothing_window)
        
    def preprocess_landmarks(self, landmarks_3d: np.ndarray) -> Optional[np.ndarray]:
        try:
            landmarks = landmarks_3d.copy()
            
            # Step 1: Recenter at wrist (landmark 0)
            wrist = landmarks[0]
            wrist_x = wrist[0]
            wrist_y = wrist[1]
            
            # Step 2: Get middle finger tip recentered distance (landmark 12)
            mid_tip = landmarks[12]
            mid_tip_x_recenter = mid_tip[0] - wrist_x
            mid_tip_y_recenter = mid_tip[1] - wrist_y
            
            # Avoid division by zero
            if mid_tip_x_recenter == 0:
                mid_tip_x_recenter = 1.0
            if mid_tip_y_recenter == 0:
                mid_tip_y_recenter = 1.0
            
            # Step 3: Apply preprocessing to all landmarks
            # Recenter, then normalize by individual x and y distances (NOT Euclidean)
            for i in range(len(landmarks)):
                landmarks[i, 0] = (landmarks[i, 0] - wrist_x) / mid_tip_x_recenter
                landmarks[i, 1] = (landmarks[i, 1] - wrist_y) / mid_tip_y_recenter
                # Keep z coordinate as is
            
            return landmarks.flatten()
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None
    
    def predict(self, landmarks_flat: np.ndarray) -> Tuple[str, float]:
        """Get gesture prediction."""
        try:
            pred_class = self.model.predict(landmarks_flat.reshape(1, -1))[0]
            
            # Get confidence
            if hasattr(self.model, 'predict_proba'):
                conf = float(np.max(self.model.predict_proba(landmarks_flat.reshape(1, -1))[0]))
            else:
                conf = 0.9
            
            # Handle string or int predictions
            if isinstance(pred_class, str):
                return pred_class, conf
            else:
                return GESTURE_CLASSES[int(pred_class)], conf
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Unknown", 0.0
    
    def get_smoothed_result(self) -> Tuple[str, float]:
        """Get smoothed prediction."""
        if not self.pred_history:
            return "Waiting", 0.0
        
        # Most common prediction
        gesture = max(set(self.pred_history), key=list(self.pred_history).count)
        conf = float(np.mean(list(self.conf_history)))
        
        return gesture, conf
    
    def draw_hand_and_gesture(self, frame: np.ndarray, landmarks, gesture: str, conf: float) -> np.ndarray:
        """Draw landmarks and gesture label."""
        h, w = frame.shape[:2]
        
        # Draw skeleton
        lm_pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        
        for connection in HAND_CONNECTIONS:
            pt1 = lm_pts[connection[0]]
            pt2 = lm_pts[connection[1]]
            cv2.line(frame, pt1, pt2, (100, 150, 255), 2)
        
        # Draw joints
        for pt in lm_pts:
            cv2.circle(frame, pt, 4, (0, 255, 0), -1)
        
        # Draw gesture label
        xs = [p[0] for p in lm_pts]
        ys = [p[1] for p in lm_pts]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        label = f"{gesture} ({conf:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        thickness = 2
        
        text_size = cv2.getTextSize(label, font, fontScale, thickness)[0]
        text_x = max(10, x_min)
        text_y = max(25, y_min - 10)
        
        # Background
        cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 8),
                     (text_x + text_size[0] + 5, text_y + 5), (0, 200, 0), -1)
        cv2.putText(frame, label, (text_x, text_y), font, fontScale, (0, 0, 0), thickness)
        
        return frame
    
    def process_video(self, video_path: str, output_path: str = None, skip_frames: int = 1) -> str:
        """
        Process video and generate inference output.
        
        Args:
            video_path: Input video file path
            output_path: Output video path (auto-generated if None)
            skip_frames: Process every nth frame (for speed)
            
        Returns:
            Path to output video
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Output path
        if output_path is None:
            output_path = video_path.parent / f"output_{video_path.name}"
        else:
            output_path = Path(output_path)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[INFO] Video: {width}x{height} @ {fps}fps, {total} frames")
        print(f"[INFO] Processing every {skip_frames} frame(s) for speed...")
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps // skip_frames, (width, height))
        
        frame_idx = 0
        processed = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                
                # Skip frames
                if frame_idx % skip_frames != 0:
                    continue
                
                processed += 1
                
                # Detect hands
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)
                
                if results.multi_hand_landmarks:
                    hand_lm = results.multi_hand_landmarks[0]
                    lm_3d = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark])
                    
                    # Preprocess and predict
                    lm_flat = self.preprocess_landmarks(lm_3d)
                    
                    if lm_flat is not None:
                        gesture, conf = self.predict(lm_flat)
                        self.pred_history.append(gesture)
                        self.conf_history.append(conf)
                        
                        # Draw
                        gesture_smooth, conf_smooth = self.get_smoothed_result()
                        if conf_smooth > 0.3:
                            frame = self.draw_hand_and_gesture(frame, hand_lm.landmark, gesture_smooth, conf_smooth)
                else:
                    self.pred_history.clear()
                    self.conf_history.clear()
                
                # Add frame counter
                cv2.putText(frame, f"Frame: {processed}/{total // skip_frames}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                # Write output
                out.write(frame)
                
                if processed % 10 == 0:
                    print(f"[INFO] Processed {processed}/{total // skip_frames} frames...")
        
        except Exception as e:
            print(f"[ERROR] {e}")
        
        finally:
            cap.release()
            out.release()
            print(f"[OK] Output saved: {output_path}")
        
        return str(output_path)


def main():
    """Main entry point."""
    PROJECT_DIR = Path(__file__).parent
    
    #change these paths to your model and video locations
    model_path = Path(r"C:\Users\Dell\Desktop\ml-project\random_forest_model.pkl")
    
    video_file = Path(r"C:\Users\Dell\Pictures\Camera Roll\WIN_20260228_13_43_04_Pro.mp4")
    

    
    # Run inference
    output_file = PROJECT_DIR / "gesture_demo.mp4"
    
    try:
        inference = GestureInference(str(model_path), smoothing_window=3)
        result_path = inference.process_video(
            str(video_file),
            str(output_file),
            skip_frames=2  # Process every 2nd frame for speed
        )
        print(f"\n[SUCCESS] Demo video created: {result_path}")
        print(f"[INFO] File size: {Path(result_path).stat().st_size / (1024*1024):.1f} MB")
    
    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
