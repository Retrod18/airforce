"""
DRDO Air Defence ML Project
Step 2: Train ML Models
  Model 1 → classify system as Traditional / Modern
  Model 2 → predict war scenario outcome (3-class)
  Model 3 → predict attacker win probability (regression)
Saves trained models + scaler to /models/ for FastAPI use
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
import joblib

warnings.filterwarnings("ignore")

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    mean_absolute_error,
    r2_score
)

# ==========================================================
# PATH CONFIGURATION (DYNAMIC — WORKS ON ANY SYSTEM)
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

systems_df = pd.read_csv(os.path.join(DATA_DIR, "air_systems_enhanced.csv"))
scenarios_df = pd.read_csv(os.path.join(DATA_DIR, "conflict_scenarios.csv"))

print("=" * 60)
print("   DRDO AIR DEFENCE — ML MODEL TRAINING")
print("=" * 60)

# ==========================================================
# MODEL 1: TECHNOLOGY CLASSIFICATION
# ==========================================================

print("\n📦 MODEL 1 – Technology Classification")

sys_type_enc = LabelEncoder()
systems_df["system_type_enc"] = sys_type_enc.fit_transform(
    systems_df["system_type"]
)

M1_FEATURES = [
    "tech_generation", "year_inducted", "stealth_rating",
    "ew_capability", "max_speed_kmph", "range_km",
    "max_altitude_m", "reliability", "cost_million_usd",
    "threat_level", "payload_kg", "system_type_enc"
]
M1_TARGET = "classification"

X1 = systems_df[M1_FEATURES]
y1 = systems_df[M1_TARGET]

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.25, random_state=42, stratify=y1
)

scaler_m1 = StandardScaler()
X1_train_sc = scaler_m1.fit_transform(X1_train)
X1_test_sc = scaler_m1.transform(X1_test)

rf1 = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced")
gb1 = GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)

rf1.fit(X1_train_sc, y1_train)
gb1.fit(X1_train_sc, y1_train)

rf1_acc = accuracy_score(y1_test, rf1.predict(X1_test_sc))
gb1_acc = accuracy_score(y1_test, gb1.predict(X1_test_sc))
model1 = rf1 if rf1_acc >= gb1_acc else gb1

cv_scores = cross_val_score(model1, X1_train_sc, y1_train, cv=StratifiedKFold(5), scoring="accuracy")

print(f"  Accuracy: {max(rf1_acc, gb1_acc):.3f}")
print(f"  CV Mean : {cv_scores.mean():.3f}")

# Save Model 1
joblib.dump(model1, os.path.join(MODEL_DIR, "model1_classifier.pkl"))
joblib.dump(scaler_m1, os.path.join(MODEL_DIR, "scaler_m1.pkl"))
joblib.dump(sys_type_enc, os.path.join(MODEL_DIR, "system_type_encoder.pkl"))

with open(os.path.join(MODEL_DIR, "model1_features.json"), "w") as fh:
    json.dump({"features": M1_FEATURES, "target": M1_TARGET}, fh)

print("  ✅ Model 1 saved")

# ==========================================================
# MODEL 2: WAR SCENARIO OUTCOME
# ==========================================================

print("\n📦 MODEL 2 – War Scenario Outcome")

M2_FEATURES = [
    "att_avg_threat", "att_avg_tech_gen", "att_modern_pct",
    "att_avg_stealth", "att_avg_ew",
    "att_fighter_count", "att_sam_count", "att_uav_count",
    "att_military_budget", "att_aircraft_count", "att_zone",
    "dfn_avg_threat", "dfn_avg_tech_gen", "dfn_modern_pct",
    "dfn_avg_stealth", "dfn_avg_ew",
    "dfn_fighter_count", "dfn_sam_count", "dfn_uav_count",
    "dfn_military_budget", "dfn_aircraft_count", "dfn_zone",
    "threat_ratio", "tech_ratio", "number_ratio", "budget_ratio"
]
M2_TARGET = "outcome"

X2 = scenarios_df[M2_FEATURES]
y2 = scenarios_df[M2_TARGET]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42, stratify=y2
)

scaler_m2 = StandardScaler()
X2_train_sc = scaler_m2.fit_transform(X2_train)
X2_test_sc = scaler_m2.transform(X2_test)

rf2 = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight="balanced")
rf2.fit(X2_train_sc, y2_train)

acc2 = accuracy_score(y2_test, rf2.predict(X2_test_sc))

print(f"  Accuracy: {acc2:.3f}")

joblib.dump(rf2, os.path.join(MODEL_DIR, "model2_war_outcome.pkl"))
joblib.dump(scaler_m2, os.path.join(MODEL_DIR, "scaler_m2.pkl"))

with open(os.path.join(MODEL_DIR, "model2_features.json"), "w") as fh:
    json.dump({"features": M2_FEATURES, "target": M2_TARGET}, fh)

print("  ✅ Model 2 saved")

# ==========================================================
# MODEL 3: WIN PROBABILITY
# ==========================================================

print("\n📦 MODEL 3 – Attacker Win Probability")

y3 = scenarios_df["attacker_win_probability"]

X3_train, X3_test, y3_train, y3_test = train_test_split(
    X2, y3, test_size=0.2, random_state=42
)

X3_train_sc = scaler_m2.transform(X3_train)
X3_test_sc = scaler_m2.transform(X3_test)

rfr = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rfr.fit(X3_train_sc, y3_train)

y3_pred = rfr.predict(X3_test_sc)

mae = mean_absolute_error(y3_test, y3_pred)
r2 = r2_score(y3_test, y3_pred)

print(f"  MAE: {mae:.4f}")
print(f"  R² : {r2:.4f}")

joblib.dump(rfr, os.path.join(MODEL_DIR, "model3_win_prob.pkl"))

print("  ✅ Model 3 saved")

# ==========================================================
# METADATA
# ==========================================================

meta = {
    "model1_accuracy": round(max(rf1_acc, gb1_acc), 4),
    "model2_accuracy": round(acc2, 4),
    "model3_r2": round(r2, 4),
    "system_types": list(sys_type_enc.classes_)
}

with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as fh:
    json.dump(meta, fh, indent=2)

print("  ✅ Metadata saved")

# ==========================================================
# SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("   TRAINING COMPLETE — ALL MODELS READY")
print("=" * 60)

print(f"\nSaved files in: {MODEL_DIR}\n")

for f in os.listdir(MODEL_DIR):
    full_path = os.path.join(MODEL_DIR, f)
    size = os.path.getsize(full_path) / 1024
    print(f"{f:<40} {size:.1f} KB")

print("\nStep 2 COMPLETE → proceed to FastAPI")
print("=" * 60)
