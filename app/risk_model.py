import os, numpy as np, joblib
import config as C
FEATURES = ["avg_speed","max_speed","density","min_gap","min_ttc","wrong_way","lane_changes","pedestrians","night"]
clip = lambda x: float(np.clip(x, 0, 1))

def rule_risk(f):
    """Expert-rule risk (0..1). Used as fallback and to bootstrap training labels."""
    a, m, d, g, t, w, lc, p, n = f
    s = (.15*clip((a-40)/50) + .05*clip((m-60)/60) + .10*clip(d/25) + .15*clip(1-g/8)
         + .20*clip((4-t)/4) + .25*min(w,1) + .10*min(lc,2)/2 + .10*clip(p/5) + .05*n)
    return clip(s*1.1)

def level(r): return "LOW" if r < .25 else "MEDIUM" if r < .5 else "HIGH" if r < .75 else "CRITICAL"

class RiskModel:
    def __init__(s):
        s.m = joblib.load(C.MODEL_PATH) if os.path.exists(C.MODEL_PATH) else None
        print("[risk] ML model loaded" if s.m else "[risk] no model found -> run train_model.py (using rules)")
    def predict(s, feats):
        x = [feats[k] for k in FEATURES]
        return clip(s.m.predict([x])[0]) if s.m else rule_risk(x)
