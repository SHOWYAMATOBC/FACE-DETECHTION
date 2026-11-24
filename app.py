import base64
import json
import io
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pathlib import Path
from proctor_ml import ProctoringEngine

app = FastAPI(
    title="AI Exam Proctoring System",
    description="5th Sem ML Project using OpenCV, NumPy, Pandas, Scikit-Learn, and FastAPI"
)

engine = ProctoringEngine(calibration_frames=50, consecutive_alert_threshold=3)
HTML_PATH = Path(__file__).parent / "static" / "index.html"

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTML_PATH.read_text(encoding="utf-8")

@app.post("/reset")
async def reset_engine():
    engine.reset()
    return JSONResponse({"message": "Engine reset and recalibrating."})

@app.get("/export-csv")
async def export_csv():
    df = engine.export_dataframe()
    if df.empty:
        return JSONResponse({"error": "No exam data logged yet."}, status_code=400)
    
    df.to_csv("exam_proctor_log.csv", index=False)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv"
    )
    response.headers["Content-Disposition"] = "attachment; filename=exam_proctor_log.csv"
    return response

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            if "," not in data:
                continue
            encoded_data = data.split(",", 1)[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            result = engine.process_frame(frame)
            await websocket.send_text(json.dumps(result))

    except WebSocketDisconnect:
        df = engine.export_dataframe()
        if not df.empty:
            df.to_csv("exam_proctor_log.csv", index=False)
            print("Session ended. Saved exam_proctor_log.csv locally.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
