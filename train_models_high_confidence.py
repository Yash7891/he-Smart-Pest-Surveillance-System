"""
Script: train_models_high_confidence.py
Trains:
  1. pest_name_predictor.joblib (Pest risk & diagnosis)
  2. pest_prediction_model_full.joblib (Pesticide recommendation)
  3. pest_prediction_model_light.joblib (Optimized for Streamlit deployment)
  4. pesticide_map.joblib
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

os.chdir("C:/Users/Om/Desktop/Smart_Pest_Surveillance")

print("📥 Loading dataset...")
df = pd.read_csv("Smart_Pesticide_MultiRecommend.csv")

features_num = ["pH", "Soil EC", "Phosphorus", "Potassium", "Urea",
                "T.S.P", "M.O.P", "Moisture", "Temperature", "Humidity"]
features_cat = ["Plant Type"]
all_features = features_num + features_cat

print(f"📊 Dataset rows: {len(df)}")
print(f"🌱 Crops: {sorted(df['Plant Type'].unique())}")
print(f"🐛 Pests: {df['pest_name'].nunique()}")
print(f"💊 Pesticides: {df['pesticide'].nunique()}")

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat)
], remainder="passthrough")

# Split
X = df[all_features]
y_pest = df["pest_name"]
y_pest_rec = df["pesticide"]

X_train, X_test, y_pest_train, y_pest_test, y_pesticide_train, y_pesticide_test = train_test_split(
    X, y_pest, y_pest_rec, test_size=0.2, random_state=42, stratify=y_pest
)

print(f"🔹 Train set: {len(X_train)} | Test set: {len(X_test)}")

# ── 1. TRAIN PEST DIAGNOSIS MODEL ──
print("\n🔬 [1/3] Training Pest Diagnosis Model...")
pest_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        random_state=42,
        n_jobs=-1
    ))
])
pest_pipe.fit(X_train, y_pest_train)
y_pest_pred = pest_pipe.predict(X_test)
pest_acc = accuracy_score(y_pest_test, y_pest_pred)
pest_conf = np.mean(np.max(pest_pipe.predict_proba(X_test), axis=1)) * 100
print(f"✅ Pest Diagnosis Model Accuracy: {pest_acc * 100:.2f}%")
print(f"✅ Pest Diagnosis Average Confidence: {pest_conf:.2f}%")
joblib.dump(pest_pipe, "pest_name_predictor.joblib", compress=1)

# ── 2. TRAIN PESTICIDE RECOMMENDATION MODEL (FULL) ──
print("\n💊 [2/3] Training Pesticide Recommendation Model (Full)...")
full_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=120,
        max_depth=18,
        random_state=42,
        n_jobs=-1
    ))
])
full_pipe.fit(X_train, y_pesticide_train)
y_pesticide_pred = full_pipe.predict(X_test)
full_acc = accuracy_score(y_pesticide_test, y_pesticide_pred)
full_conf = np.mean(np.max(full_pipe.predict_proba(X_test), axis=1)) * 100
print(f"✅ Full Pesticide Model Accuracy: {full_acc * 100:.2f}%")
print(f"✅ Full Pesticide Model Average Confidence: {full_conf:.2f}%")
joblib.dump(full_pipe, "pest_prediction_model_full.joblib", compress=1)

# ── 3. TRAIN LIGHT MODEL (FOR FAST STREAMLIT RESPONSE) ──
print("\n⚡ [3/3] Training Light Pesticide Recommendation Model...")
light_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=60,
        max_depth=14,
        random_state=42,
        n_jobs=-1
    ))
])
light_pipe.fit(X_train, y_pesticide_train)
y_light_pred = light_pipe.predict(X_test)
light_acc = accuracy_score(y_pesticide_test, y_light_pred)
light_conf = np.mean(np.max(light_pipe.predict_proba(X_test), axis=1)) * 100
print(f"✅ Light Pesticide Model Accuracy: {light_acc * 100:.2f}%")
print(f"✅ Light Pesticide Model Average Confidence: {light_conf:.2f}%")
joblib.dump(light_pipe, "pest_prediction_model_light.joblib", compress=1)

# Save pesticide map
pesticide_map = {name: idx for idx, name in enumerate(full_pipe.classes_)}
joblib.dump(pesticide_map, "pesticide_map.joblib")

print("\n" + "=" * 60)
print("🎯 TRAINING COMPLETE - SUMMARY")
print("=" * 60)
print(f"Pest Diagnosis Accuracy:       {pest_acc * 100:.2f}%  (Avg Conf: {pest_conf:.2f}%)")
print(f"Pesticide Rec (Full) Accuracy: {full_acc * 100:.2f}%  (Avg Conf: {full_conf:.2f}%)")
print(f"Pesticide Rec (Light) Accuracy:{light_acc * 100:.2f}%  (Avg Conf: {light_conf:.2f}%)")
print("=" * 60)

