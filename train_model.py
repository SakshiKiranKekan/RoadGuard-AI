"""Trains the accident-risk model. Bootstraps from simulated feature vectors labelled with expert
rules + noise. For a real deployment, replace with labelled real data (see README)."""
import sys, os, numpy as np, joblib
sys.path.insert(0, "app")
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from risk_model import rule_risk
rng = np.random.default_rng(42); N = 30000
X = np.c_[rng.gamma(4,9,N), rng.gamma(4,13,N), rng.integers(0,40,N), rng.gamma(2,4,N), rng.uniform(0.3,10,N),
          (rng.random(N)<.06).astype(int), rng.poisson(.4,N), rng.poisson(1.2,N), (rng.random(N)<.3).astype(int)]
y = np.array([rule_risk(r) for r in X]) + rng.normal(0, .04, N)
y = np.clip(y, 0, 1)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=1)
rf = RandomForestRegressor(250, min_samples_leaf=3, n_jobs=-1, random_state=1).fit(Xtr, ytr)
p = rf.predict(Xte); print(f"R2={r2_score(yte,p):.3f}  MAE={mean_absolute_error(yte,p):.3f}")
os.makedirs("models", exist_ok=True); joblib.dump(rf, "models/risk_model.joblib"); print("saved models/risk_model.joblib")
