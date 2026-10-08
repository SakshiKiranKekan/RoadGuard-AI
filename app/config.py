# ---- Edit these for YOUR camera (calibration matters for speed/distance) ----
YOLO_WEIGHTS = "yolo11n.pt"        # auto-downloaded on first run (use yolo11s.pt for accuracy)
DEFAULT_SOURCE = "0"               # "0" webcam | "rtsp://..." CCTV | "path/to/video.mp4"
CONF = 0.35
PIXELS_PER_METER = 12.0            # measure a known length (e.g. lane width 3.5 m) in pixels
ALLOWED_DIRECTION = (1.0, 0.0)     # permitted traffic direction in image (x,y): (1,0)=left->right
LANE_SHIFT_PX = 45                 # lateral shift over ~1s that counts as a lane change
SPEED_LIMIT_KMH = 60
EVENT_RISK = 0.5                   # log an event when risk >= this
EVENT_COOLDOWN_S = 15
ZONE_NAME = "Camera Zone 1"
MODEL_PATH = "models/risk_model.joblib"
DB_PATH = "roadguard.db"
