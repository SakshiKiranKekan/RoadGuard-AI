# RoadGuard AI – Real-Time Road Accident Risk Prediction (YOLO11 + OpenCV + ML)

Pipeline: camera/video → OpenCV → YOLO11 detection → ByteTrack tracking → feature extraction
(speed, gap, time-to-collision, wrong-way, lane change, density, pedestrians, night) → Random Forest risk model
→ Low/Medium/High/Critical + alerts → SQLite history/events → live web dashboard (login/signup).

## Run
```
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train_model.py        # creates models/risk_model.joblib
python run.py                # open http://localhost:8000, sign up, done
```
Source: edit `DEFAULT_SOURCE` in `app/config.py` (0 = webcam, RTSP URL, or video file), or use the
dashboard's "Set source" / "Upload video" buttons at runtime.

## IMPORTANT: calibrate for your camera (`app/config.py`)
- `PIXELS_PER_METER`: measure a known length (lane width ≈ 3.5 m) in the 960px-wide frame. Speed/gap accuracy depends on it.
- `ALLOWED_DIRECTION`: permitted traffic direction in image coordinates, e.g. (1,0) left→right, (0,1) top→bottom.
- Speeds are estimates (single camera, no perspective correction) – treat as indicative.

## Files
- `app/vision.py` – detection, tracking, speed, TTC, wrong-way, lane change, risk, MJPEG frames
- `app/risk_model.py`, `train_model.py` – feature set, rule-based fallback, Random Forest training
- `app/main.py` – FastAPI (auth, stats, events, history, upload, /video_feed)
- `app/db.py` – SQLite (users with PBKDF2 hashes, sessions, events, history)
- `app/static/index.html` – login/signup + live dashboard

## Honest notes for your viva/report
- The shipped model is bootstrapped on simulated features labelled by expert rules (no public dataset gives
  "risk" labels per frame). To make it truly learned: log feature vectors, label near-miss/accident clips
  (e.g. from public accident video datasets such as CADP / DoTA), retrain with XGBoost, and report precision/recall.
- Future work: homography for true speed, accident-detection classifier, heatmap per zone, SMS/email alerts, weather API, Docker.
