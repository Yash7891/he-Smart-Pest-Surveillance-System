"""
Retrain with SMART strategy for high confidence.

Key insight from data analysis:
- For 70% of data, pest_name='unknown_pest' → pesticide is essentially random noise
- For known pests, each pest maps to a GROUP of pesticides (e.g., stem_borer → Imidacloprid/Chlorpyrifos/etc.)
- Same environmental conditions → same pest name → same pesticide group

Strategy:
1. Train on the FULL dataset but predict pest GROUP (cluster of similar pesticides per crop+pest combo)
2. Use pest_name as a FEATURE when predicting pesticides (huge signal)
3. For the 'unknown_pest' rows, use environmental features to predict the most likely pest first
"""
import pandas as pd
import numpy as np
import joblib
import os
import gc
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ==============================================================
# 1. Load & Clean Dataset
# ==============================================================
print("📥 Loading dataset...")
df = pd.read_csv("Smart_Pesticide_MultiRecommend.csv")
df["pesticide"] = df["pesticide"].astype(str).apply(lambda x: x.split(",")[0].strip())

print(f"✅ Dataset: {len(df)} rows")
print(f"🌱 Crops: {sorted(df['Plant Type'].unique())}")
print(f"🐛 Pest names: {sorted(df['pest_name'].unique())}")
print(f"💊 Pesticide classes: {df['pesticide'].nunique()}")

# ==============================================================
# 2. STAGE 1: Pest Prediction Model (env features → pest_name)
# ==============================================================
print("\n" + "="*60)
print("🔬 STAGE 1: Training PEST PREDICTION model")
print("="*60)

# Use all rows to learn pest patterns
# For unknown_pest rows, we'll drop them from pest model training
known_df = df[df["pest_name"] != "unknown_pest"].copy()
print(f"📊 Known pest rows: {len(known_df)}")
print(f"📊 Pest classes: {known_df['pest_name'].nunique()}")

pest_features = ["pH", "Soil EC", "Phosphorus", "Potassium", "Urea",
                  "T.S.P", "M.O.P", "Moisture", "Temperature", "Humidity"]
pest_cat_features = ["Plant Type"]

pest_preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), pest_cat_features)
], remainder="passthrough")

X_pest = known_df[pest_features + pest_cat_features]
y_pest = known_df["pest_name"]

X_pest_train, X_pest_test, y_pest_train, y_pest_test = train_test_split(
    X_pest, y_pest, test_size=0.2, random_state=42, stratify=y_pest
)

pest_model = Pipeline([
    ("preprocessor", pest_preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=18,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

pest_model.fit(X_pest_train, y_pest_train)
y_pest_pred = pest_model.predict(X_pest_test)
pest_acc = accuracy_score(y_pest_test, y_pest_pred)
pest_proba = pest_model.predict_proba(X_pest_test)
pest_avg_conf = np.mean(np.max(pest_proba, axis=1)) * 100

print(f"✅ Pest Model Accuracy: {pest_acc*100:.2f}%")
print(f"✅ Pest Model Avg Confidence: {pest_avg_conf:.2f}%")

# ==============================================================
# 3. STAGE 2: Pesticide Model (env features + pest_name → pesticide)
# ==============================================================
print("\n" + "="*60)
print("💊 STAGE 2: Training PESTICIDE RECOMMENDATION model")
print("="*60)

# Now use pest_name as a feature! This is the biggest signal.
all_features = pest_features + ["Plant Type", "pest_name"]

# For unknown_pest: first predict pest, then predict pesticide
# For training: use all data but add predicted pest for unknown rows

# Predict pest for unknown rows
unknown_df = df[df["pest_name"] == "unknown_pest"].copy()
if len(unknown_df) > 0:
    unknown_pred = pest_model.predict(unknown_df[pest_features + pest_cat_features])
    unknown_df["pest_name"] = unknown_pred
    print(f"✅ Predicted pest for {len(unknown_df)} unknown rows")

# Combine known + predicted-unknown
full_df = pd.concat([known_df, unknown_df], ignore_index=True)
print(f"📊 Full training set: {len(full_df)} rows")

pesticide_preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["Plant Type", "pest_name"])
], remainder="passthrough")

X_full = full_df[all_features]
y_full = full_df["pesticide"]

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

# FULL Model
print("\n🤖 Training FULL pesticide model...")
full_pipe = Pipeline([
    ("preprocessor", pesticide_preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

full_pipe.fit(X_train, y_train)
y_pred = full_pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)
proba = full_pipe.predict_proba(X_test)
avg_conf = np.mean(np.max(proba, axis=1)) * 100

print(f"✅ FULL Model Accuracy: {acc*100:.2f}%")
print(f"✅ FULL Model Avg Confidence: {avg_conf:.2f}%")

cv_score = cross_val_score(full_pipe, X_full, y_full, cv=5, scoring="accuracy", n_jobs=-1).mean()
print(f"✅ FULL Model Cross-Val: {cv_score*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(full_pipe, "pest_prediction_model_full.joblib", compress=1)
print("💾 Saved: pest_prediction_model_full.joblib")

# LIGHT Model
print("\n💡 Training LIGHT pesticide model...")
light_pipe = Pipeline([
    ("preprocessor", pesticide_preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

light_pipe.fit(X_train, y_train)
y_pred_light = light_pipe.predict(X_test)
acc_light = accuracy_score(y_test, y_pred_light)
proba_light = light_pipe.predict_proba(X_test)
avg_conf_light = np.mean(np.max(proba_light, axis=1)) * 100

print(f"✅ LIGHT Model Accuracy: {acc_light*100:.2f}%")
print(f"✅ LIGHT Model Avg Confidence: {avg_conf_light:.2f}%")

joblib.dump(light_pipe, "pest_prediction_model_light.joblib", compress=1)
print("💾 Saved: pest_prediction_model_light.joblib")

# Save pest model separately
joblib.dump(pest_model, "pest_name_predictor.joblib", compress=1)
print("💾 Saved: pest_name_predictor.joblib")

# Save pesticide map
pesticide_map = {name: idx for idx, name in enumerate(full_pipe.classes_)}
joblib.dump(pesticide_map, "pesticide_map.joblib")
print("💾 Saved: pesticide_map.joblib")

# ==============================================================
# SUMMARY
# ==============================================================
print("\n" + "="*60)
print("📊 FINAL TRAINING SUMMARY")
print("="*60)
print(f"  Dataset rows:             {len(df)}")
print(f"  Known pest rows:          {len(known_df)}")
print(f"  Features (pesticide):     {len(all_features)} (incl. pest_name)")
print(f"  Pesticide classes:        {y_full.nunique()}")
print(f"  ──────────────────────────────────────")
print(f"  🐛 PEST MODEL:")
print(f"     Accuracy:              {pest_acc*100:.2f}%")
print(f"     Avg Confidence:        {pest_avg_conf:.2f}%")
print(f"  ──────────────────────────────────────")
print(f"  💊 FULL PESTICIDE MODEL:")
print(f"     Accuracy:              {acc*100:.2f}%")
print(f"     Avg Confidence:        {avg_conf:.2f}%")
print(f"     Cross-Val:             {cv_score*100:.2f}%")
print(f"  ──────────────────────────────────────")
print(f"  💊 LIGHT PESTICIDE MODEL:")
print(f"     Accuracy:              {acc_light*100:.2f}%")
print(f"     Avg Confidence:        {avg_conf_light:.2f}%")
print("="*60)

gc.collect()
print("\n🎉 Done!")
