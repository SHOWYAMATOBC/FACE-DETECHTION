# Online Exam Movement & Anomaly Detection System
**5th Semester Machine Learning Final Project**

## Tech Stack
- **OpenCV (`cv2`)**: Real-time Haar Cascade spatial localization (face & eye detection) and Gaussian-smoothed temporal frame differencing.
- **NumPy**: High-speed matrix operations, Euclidean distance calculation, and frame buffer decoding.
- **Pandas**: Time-series telemetry logging, rolling median baseline calculation, and automated CSV report generation.
- **Scikit-Learn**: `StandardScaler` feature normalization + `IsolationForest` unsupervised outlier detection to flag abnormal posture or sudden movement bursts.
- **FastAPI**: Asynchronous WebSocket streaming server and REST endpoints for recalibration and CSV report downloads.

## Project Structure
```
ml_proctoring_system/
├── app.py              # FastAPI server, WebSocket handler & CSV export endpoints
├── proctor_ml.py       # Feature extraction & IsolationForest ML pipeline
├── static/
│   └── index.html      # Live proctoring dashboard UI
├── requirements.txt    # Python dependencies
└── README.md           # Setup and execution guide
```

## How to Run Locally

1. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the FastAPI server:**
   ```bash
   uvicorn app:app --reload
   ```

4. **Open the Proctoring Dashboard:**
   Navigate to `http://localhost:8000` in your browser and allow webcam permissions.

## How the ML Pipeline Works
1. **Calibration Phase (First 50 Frames):** Sit naturally facing the screen. The system collects your baseline facial coordinates (`face_x`, `face_y`), bounding box scale (`face_area`), and micro-movements (`movement_score`).
2. **Unsupervised Training:** `StandardScaler` normalizes the 50 calibration frames and fits an `IsolationForest` model specifically tuned to your posture and camera setup.
3. **Real-Time Inference & Debouncing:** Every incoming frame is scored. A rolling buffer of 3 consecutive anomalous frames is required before triggering a visual violation alert, preventing false positives from normal blinking or minor adjustments.
