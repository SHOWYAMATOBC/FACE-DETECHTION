import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import datetime

class ProctoringEngine:
    def __init__(self, calibration_frames: int = 50, consecutive_alert_threshold: int = 3):
        self.calibration_frames = calibration_frames
        self.consecutive_alert_threshold = consecutive_alert_threshold
        
        # Haar Cascades from OpenCV
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Scikit-Learn Pipeline Components
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.06,
            random_state=42
        )
        
        self.reset()

    def reset(self):
        self.prev_gray = None
        self.frame_count = 0
        self.is_trained = False
        self.baseline_x = 320.0
        self.baseline_y = 240.0
        self.consecutive_anomalies = 0
        self.total_violations = 0
        self.exam_log = []

    def process_frame(self, frame: np.ndarray) -> dict:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 1. Spatial Feature Extraction (Face & Eyes via Haar Cascades)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
        )
        num_faces = len(faces)
        num_eyes = 0
        bbox = []

        face_x, face_y, face_area = self.baseline_x, self.baseline_y, 0.0

        if num_faces == 1:
            x, y, w, h = faces[0]
            bbox = [int(x), int(y), int(w), int(h)]
            face_x = float(x + w / 2.0)
            face_y = float(y + h / 2.0)
            face_area = float(w * h)

            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4)
            num_eyes = len(eyes)

        # 2. Temporal Feature Extraction (Motion Velocity via Frame Differencing)
        movement_score = 0.0
        if self.prev_gray is not None:
            diff = cv2.absdiff(self.prev_gray, gray_blurred)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            movement_score = float(np.sum(thresh) / 1000.0) # Scaled pixel change velocity
        self.prev_gray = gray_blurred

        # Calculate Euclidean head offset from calibrated center
        head_offset = float(np.hypot(face_x - self.baseline_x, face_y - self.baseline_y))

        # 3. Rule-Based Immediate Flags (Missing Face / Multiple People)
        raw_status = "OK"
        if num_faces == 0:
            raw_status = "ALERT: NO FACE DETECTED"
        elif num_faces > 1:
            raw_status = "ALERT: MULTIPLE FACES DETECTED"

        # 4. Unsupervised ML Anomaly Detection (Isolation Forest)
        anomaly_score = 0.0
        feature_dict = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "frame": self.frame_count,
            "num_faces": num_faces,
            "num_eyes": num_eyes,
            "movement_score": round(movement_score, 3),
            "face_x": round(face_x, 2),
            "face_y": round(face_y, 2),
            "face_area": round(face_area, 2),
            "head_offset": round(head_offset, 2),
        }

        self.exam_log.append(feature_dict)
        self.frame_count += 1

        feature_cols = ["movement_score", "face_x", "face_y", "face_area", "head_offset"]

        # Train or periodically update baseline
        if self.frame_count >= self.calibration_frames and not self.is_trained:
            df_calib = pd.DataFrame(self.exam_log[:self.calibration_frames])
            # Update neutral head center from calibration median
            self.baseline_x = float(df_calib["face_x"].median())
            self.baseline_y = float(df_calib["face_y"].median())
            
            X_train = self.scaler.fit_transform(df_calib[feature_cols].values)
            self.model.fit(X_train)
            self.is_trained = True

        if self.is_trained:
            x_curr = np.array([[movement_score, face_x, face_y, face_area, head_offset]])
            x_scaled = self.scaler.transform(x_curr)
            pred = int(self.model.predict(x_scaled)[0]) # 1 = normal, -1 = anomaly
            anomaly_score = float(self.model.decision_function(x_scaled)[0])

            if raw_status == "OK" and pred == -1:
                raw_status = "ALERT: ANOMALOUS HEAD/BODY MOVEMENT"

        # 5. Rolling Buffer Thresholding (Prevents single-frame flicker)
        if raw_status != "OK":
            self.consecutive_anomalies += 1
        else:
            self.consecutive_anomalies = 0

        final_status = "OK"
        if self.consecutive_anomalies >= self.consecutive_alert_threshold:
            final_status = raw_status
            self.total_violations += 1

        self.exam_log[-1]["status"] = final_status
        self.exam_log[-1]["anomaly_score"] = round(anomaly_score, 4)

        return {
            "status": final_status,
            "num_faces": num_faces,
            "num_eyes": num_eyes,
            "movement_score": movement_score,
            "head_offset": head_offset,
            "anomaly_score": anomaly_score,
            "model_trained": self.is_trained,
            "frame_count": self.frame_count,
            "total_violations": self.total_violations,
            "bbox": bbox
        }

    def export_dataframe(self) -> pd.DataFrame:
        if not self.exam_log:
            return pd.DataFrame()
        return pd.DataFrame(self.exam_log)
