import cv2, time, threading, math, datetime
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
import config as C, db
from risk_model import RiskModel, level

PED = {0}   # COCO: 0 person | 1 bicycle 2 car 3 motorcycle 5 bus 7 truck
AX, AY = C.ALLOWED_DIRECTION; _n = math.hypot(AX, AY); AX, AY = AX/_n, AY/_n

class Pipeline:
    def __init__(s):
        s.yolo = YOLO(C.YOLO_WEIGHTS); s.rm = RiskModel()
        s.jpeg = None; s.stats = {"risk": 0, "level": "LOW", "vehicles": 0, "pedestrians": 0, "avg_speed": 0,
                                  "density": "Low", "min_gap": None, "min_ttc": None, "alerts": [], "fps": 0}
        s.thread = None; s.stop = threading.Event(); s.risk = 0.0; s.last_ev = 0; s.last_hist = 0

    def start(s, src=None):
        if s.thread and s.thread.is_alive(): s.stop.set(); s.thread.join(timeout=5)
        s.stop = threading.Event(); s.src = src or C.DEFAULT_SOURCE
        s.thread = threading.Thread(target=s._run, daemon=True); s.thread.start()

    def _run(s):
        src = int(s.src) if str(s.src).isdigit() else s.src
        cap = cv2.VideoCapture(src); is_file = isinstance(src, str) and not src.startswith(("rtsp", "http"))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30; fno = 0; tr = defaultdict(lambda: deque(maxlen=int(fps)))
        s.yolo.predictor = None   # reset tracker state for new source
        while not s.stop.is_set():
            t0 = time.time(); ok, frame = cap.read()
            if not ok:
                if is_file: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                time.sleep(.5); cap.release(); cap = cv2.VideoCapture(src); continue
            fno += 1; now = fno / fps
            frame = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))
            r = s.yolo.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0,1,2,3,5,7], conf=C.CONF, verbose=False)[0]
            objs = []
            if r.boxes is not None and r.boxes.id is not None:
                for b, i, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().cpu().tolist(), r.boxes.cls.int().cpu().tolist()):
                    objs.append((i, c, b))
            veh = {}; peds = 0
            for i, c, b in objs:
                cx_, cy_ = (b[0]+b[2])/2, (b[1]+b[3])/2
                if c in PED: peds += 1; continue
                tr[i].append((now, cx_, cy_)); h = tr[i]; sp = 0; vx = vy = 0; wrong = lane = False
                if len(h) >= 6:
                    dt = h[-1][0] - h[0][0] or 1e-6; dx, dy = h[-1][1]-h[0][1], h[-1][2]-h[0][2]
                    vx, vy = dx/dt, dy/dt; sp = math.hypot(vx, vy) / C.PIXELS_PER_METER * 3.6
                    wrong = (vx*AX + vy*AY) < 0 and sp > 8
                    lane = abs(dx*(-AY) + dy*AX) > C.LANE_SHIFT_PX and (dx*AX + dy*AY) > 0 and len(h) > fps*.6
                veh[i] = dict(b=b, c=(cx_, cy_), v=(vx, vy), sp=sp, wrong=wrong, lane=lane)
            min_gap, min_ttc, risky = 99.0, 10.0, set(); ids = list(veh)   # pairwise gap + time-to-collision
            for a in range(len(ids)):
                for k in range(a+1, len(ids)):
                    A, B = veh[ids[a]], veh[ids[k]]
                    gx = max(0, max(B["b"][0]-A["b"][2], A["b"][0]-B["b"][2])); gy = max(0, max(B["b"][1]-A["b"][3], A["b"][1]-B["b"][3]))
                    gap = math.hypot(gx, gy) / C.PIXELS_PER_METER
                    if gap > 25: continue
                    px, py = B["c"][0]-A["c"][0], B["c"][1]-A["c"][1]; d = math.hypot(px, py) or 1
                    closing = -((px*(B["v"][0]-A["v"][0]) + py*(B["v"][1]-A["v"][1])) / d)
                    ttc = gap*C.PIXELS_PER_METER/closing if closing > 5 else 10.0
                    min_gap = min(min_gap, gap); min_ttc = min(min_ttc, ttc)
                    if gap < 2.5 or ttc < 2.5: risky |= {ids[a], ids[k]}
            sps = [v["sp"] for v in veh.values()]; n = len(veh); hr = datetime.datetime.now().hour
            f = dict(avg_speed=float(np.mean(sps)) if sps else 0, max_speed=max(sps, default=0), density=n,
                     min_gap=min_gap, min_ttc=min_ttc, wrong_way=int(any(v["wrong"] for v in veh.values())),
                     lane_changes=sum(v["lane"] for v in veh.values()), pedestrians=peds, night=int(hr < 6 or hr >= 19))
            s.risk += (s.rm.predict(f) - s.risk) * .15; lv = level(s.risk); alerts = []
            if f["wrong_way"]: alerts.append("Wrong-way vehicle")
            if min_gap < 2.5: alerts.append("Unsafe following distance")
            if min_ttc < 2.5: alerts.append("Potential collision (TTC %.1fs)" % min_ttc)
            if f["max_speed"] > C.SPEED_LIMIT_KMH: alerts.append("Overspeeding")
            if f["lane_changes"]: alerts.append("Sudden lane change")
            if peds >= 3: alerts.append("Pedestrian crowd on road")
            col = {"LOW": (80,200,80), "MEDIUM": (0,200,230), "HIGH": (0,140,255), "CRITICAL": (60,60,240)}[lv]
            for i, v in veh.items():
                b = v["b"].astype(int); bad = v["wrong"] or v["lane"] or i in risky or v["sp"] > C.SPEED_LIMIT_KMH
                cc = (60,60,240) if bad else (80,200,80)
                cv2.rectangle(frame, tuple(b[:2]), tuple(b[2:]), cc, 2)
                cv2.putText(frame, f"#{i} {v['sp']:.0f}km/h" + (" WRONG WAY" if v["wrong"] else ""), (b[0], max(14, b[1]-5)), 0, .5, cc, 2)
            for i, c, b in objs:
                if c in PED: b = b.astype(int); cv2.rectangle(frame, tuple(b[:2]), tuple(b[2:]), (0,215,255), 2)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (20, 20, 30), -1)
            cv2.putText(frame, f"RISK {s.risk*100:.0f}% {lv} | veh {n} ped {peds}", (10, 24), 0, .75, col, 2)
            s.jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            s.stats = dict(risk=round(s.risk, 3), level=lv, vehicles=n, pedestrians=peds, avg_speed=round(f["avg_speed"], 1),
                           density="Low" if n < 6 else "Moderate" if n < 12 else "Heavy" if n < 20 else "Severe",
                           min_gap=None if min_gap == 99 else round(min_gap, 1), min_ttc=None if min_ttc >= 10 else round(min_ttc, 1),
                           alerts=alerts, fps=round(1/max(time.time()-t0, 1e-3), 1))
            t = time.time()
            if t - s.last_hist > 2: db.add_history(s.risk, n, f["avg_speed"]); s.last_hist = t
            if s.risk >= C.EVENT_RISK and t - s.last_ev > C.EVENT_COOLDOWN_S:
                db.add_event(s.risk, lv, "; ".join(alerts) or "Elevated risk"); s.last_ev = t
            if is_file: time.sleep(max(0, 1/fps - (time.time()-t0)))
        cap.release()
