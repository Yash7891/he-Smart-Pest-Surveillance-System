"""
Script: generate_high_quality_dataset.py
Purpose: Synthesize a high-precision, agronomy-grounded dataset for
         Smart Pest Surveillance & Pesticide Recommendation.

Eliminates uniform noise by grounding each pest and pesticide in distinct,
scientifically valid environmental (Temp, Humidity, Moisture) and soil (pH, EC, NPK) profiles.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# Define agronomic knowledge base for all 10 crops
PROFILES = [
    # ── 1. RICE ──
    {
        "plant": "rice", "pest": "Brown Spot (Fungal)", "pesticide": "Propiconazole", "alt_pesticide": "Validamycin",
        "temp": (27.0, 2.0), "hum": (88.0, 4.0), "moist": (75.0, 5.0),
        "ph": (5.8, 0.3), "ec": (0.6, 0.15), "p": (35.0, 8.0), "k": (160.0, 20.0), "urea": (110.0, 15.0), "tsp": (45.0, 8.0), "mop": (40.0, 8.0),
        "count": 4000
    },
    {
        "plant": "rice", "pest": "Yellow Stem Borer", "pesticide": "Chlorantraniliprole", "alt_pesticide": "Chlorpyrifos",
        "temp": (31.0, 2.2), "hum": (78.0, 5.0), "moist": (65.0, 6.0),
        "ph": (6.4, 0.3), "ec": (0.7, 0.15), "p": (50.0, 10.0), "k": (220.0, 25.0), "urea": (140.0, 15.0), "tsp": (55.0, 8.0), "mop": (50.0, 8.0),
        "count": 4000
    },
    {
        "plant": "rice", "pest": "Rice Leaf Folder", "pesticide": "Cartap Hydrochloride", "alt_pesticide": "Cypermethrin",
        "temp": (28.5, 2.0), "hum": (84.0, 4.5), "moist": (70.0, 5.5),
        "ph": (6.2, 0.3), "ec": (0.65, 0.15), "p": (45.0, 8.0), "k": (200.0, 20.0), "urea": (130.0, 12.0), "tsp": (50.0, 8.0), "mop": (45.0, 8.0),
        "count": 4000
    },
    {
        "plant": "rice", "pest": "Bacterial Blight", "pesticide": "Copper Oxychloride", "alt_pesticide": "Streptocycline",
        "temp": (32.0, 1.8), "hum": (92.0, 3.5), "moist": (85.0, 4.5),
        "ph": (6.0, 0.25), "ec": (0.8, 0.15), "p": (40.0, 8.0), "k": (180.0, 20.0), "urea": (125.0, 15.0), "tsp": (50.0, 8.0), "mop": (45.0, 8.0),
        "count": 4000
    },

    # ── 2. WHEAT ──
    {
        "plant": "wheat", "pest": "Stripe / Brown Rust", "pesticide": "Propiconazole", "alt_pesticide": "Tebuconazole",
        "temp": (19.0, 2.5), "hum": (86.0, 4.5), "moist": (68.0, 5.0),
        "ph": (6.5, 0.3), "ec": (0.55, 0.12), "p": (45.0, 8.0), "k": (210.0, 20.0), "urea": (85.0, 10.0), "tsp": (40.0, 7.0), "mop": (45.0, 7.0),
        "count": 4000
    },
    {
        "plant": "wheat", "pest": "Wheat Aphids", "pesticide": "Imidacloprid", "alt_pesticide": "Thiamethoxam",
        "temp": (23.0, 2.5), "hum": (52.0, 6.0), "moist": (42.0, 5.5),
        "ph": (6.7, 0.3), "ec": (0.6, 0.12), "p": (55.0, 10.0), "k": (240.0, 25.0), "urea": (95.0, 12.0), "tsp": (45.0, 8.0), "mop": (50.0, 8.0),
        "count": 4000
    },
    {
        "plant": "wheat", "pest": "Armyworm", "pesticide": "Emamectin Benzoate", "alt_pesticide": "Chlorpyrifos",
        "temp": (28.0, 2.5), "hum": (62.0, 5.5), "moist": (52.0, 6.0),
        "ph": (6.6, 0.3), "ec": (0.65, 0.15), "p": (50.0, 9.0), "k": (230.0, 22.0), "urea": (105.0, 14.0), "tsp": (50.0, 8.0), "mop": (50.0, 8.0),
        "count": 4000
    },

    # ── 3. TOMATO ──
    {
        "plant": "tomato", "pest": "Early / Late Blight", "pesticide": "Mancozeb", "alt_pesticide": "Copper Oxychloride",
        "temp": (22.0, 2.2), "hum": (89.0, 4.0), "moist": (78.0, 5.0),
        "ph": (6.1, 0.3), "ec": (0.75, 0.15), "p": (60.0, 10.0), "k": (260.0, 25.0), "urea": (90.0, 12.0), "tsp": (55.0, 8.0), "mop": (60.0, 9.0),
        "count": 4000
    },
    {
        "plant": "tomato", "pest": "Tomato Fruit Borer", "pesticide": "Chlorantraniliprole", "alt_pesticide": "Flubendiamide",
        "temp": (29.5, 2.4), "hum": (66.0, 5.5), "moist": (56.0, 6.0),
        "ph": (6.5, 0.3), "ec": (0.7, 0.15), "p": (65.0, 10.0), "k": (280.0, 25.0), "urea": (100.0, 12.0), "tsp": (60.0, 8.0), "mop": (65.0, 9.0),
        "count": 4000
    },
    {
        "plant": "tomato", "pest": "Whitefly & Leaf Curl", "pesticide": "Acetamiprid", "alt_pesticide": "Imidacloprid",
        "temp": (34.0, 2.5), "hum": (44.0, 6.0), "moist": (38.0, 5.5),
        "ph": (6.8, 0.3), "ec": (0.8, 0.15), "p": (55.0, 9.0), "k": (250.0, 25.0), "urea": (85.0, 10.0), "tsp": (50.0, 8.0), "mop": (55.0, 8.0),
        "count": 4000
    },

    # ── 4. CORN (MAIZE) ──
    {
        "plant": "corn", "pest": "Fall Armyworm", "pesticide": "Emamectin Benzoate", "alt_pesticide": "Chlorantraniliprole",
        "temp": (30.0, 2.5), "hum": (64.0, 5.5), "moist": (54.0, 6.0),
        "ph": (6.4, 0.3), "ec": (0.7, 0.15), "p": (50.0, 9.0), "k": (210.0, 22.0), "urea": (135.0, 15.0), "tsp": (55.0, 8.0), "mop": (50.0, 8.0),
        "count": 4000
    },
    {
        "plant": "corn", "pest": "Maize Stem Borer", "pesticide": "Chlorpyrifos", "alt_pesticide": "Carbofuran",
        "temp": (27.0, 2.2), "hum": (72.0, 5.0), "moist": (62.0, 5.5),
        "ph": (6.3, 0.3), "ec": (0.65, 0.15), "p": (45.0, 8.0), "k": (200.0, 20.0), "urea": (120.0, 14.0), "tsp": (50.0, 8.0), "mop": (45.0, 8.0),
        "count": 4000
    },
    {
        "plant": "corn", "pest": "Maydis Leaf Blight", "pesticide": "Mancozeb", "alt_pesticide": "Azoxystrobin",
        "temp": (28.0, 2.0), "hum": (88.0, 4.0), "moist": (76.0, 5.0),
        "ph": (6.2, 0.25), "ec": (0.75, 0.15), "p": (48.0, 8.0), "k": (195.0, 20.0), "urea": (115.0, 12.0), "tsp": (48.0, 7.0), "mop": (45.0, 7.0),
        "count": 4000
    },

    # ── 5. CHILI ──
    {
        "plant": "chili", "pest": "Chili Thrips & Mites", "pesticide": "Abamectin", "alt_pesticide": "Diafenthiuron",
        "temp": (34.5, 2.4), "hum": (42.0, 5.5), "moist": (36.0, 5.0),
        "ph": (6.6, 0.3), "ec": (0.8, 0.18), "p": (60.0, 10.0), "k": (260.0, 25.0), "urea": (95.0, 12.0), "tsp": (55.0, 8.0), "mop": (60.0, 8.0),
        "count": 4000
    },
    {
        "plant": "chili", "pest": "Anthracnose / Dieback", "pesticide": "Azoxystrobin", "alt_pesticide": "Carbendazim",
        "temp": (29.0, 2.2), "hum": (87.0, 4.2), "moist": (74.0, 5.2),
        "ph": (6.3, 0.3), "ec": (0.7, 0.15), "p": (55.0, 9.0), "k": (240.0, 22.0), "urea": (90.0, 10.0), "tsp": (50.0, 8.0), "mop": (55.0, 8.0),
        "count": 4000
    },
    {
        "plant": "chili", "pest": "Chili Aphids", "pesticide": "Thiamethoxam", "alt_pesticide": "Imidacloprid",
        "temp": (25.0, 2.2), "hum": (58.0, 5.5), "moist": (48.0, 5.0),
        "ph": (6.5, 0.25), "ec": (0.75, 0.15), "p": (58.0, 9.0), "k": (250.0, 22.0), "urea": (85.0, 10.0), "tsp": (52.0, 8.0), "mop": (58.0, 8.0),
        "count": 4000
    },

    # ── 6. EGGPLANT ──
    {
        "plant": "eggplant", "pest": "Shoot and Fruit Borer", "pesticide": "Cypermethrin", "alt_pesticide": "Chlorantraniliprole",
        "temp": (31.0, 2.5), "hum": (74.0, 5.5), "moist": (64.0, 6.0),
        "ph": (6.3, 0.3), "ec": (0.75, 0.15), "p": (62.0, 10.0), "k": (270.0, 25.0), "urea": (110.0, 14.0), "tsp": (58.0, 8.0), "mop": (62.0, 9.0),
        "count": 4000
    },
    {
        "plant": "eggplant", "pest": "Bacterial Wilt", "pesticide": "Copper Oxychloride", "alt_pesticide": "Streptocycline",
        "temp": (32.5, 2.0), "hum": (90.0, 3.8), "moist": (88.0, 4.0),
        "ph": (6.0, 0.25), "ec": (0.85, 0.18), "p": (50.0, 8.0), "k": (220.0, 22.0), "urea": (95.0, 12.0), "tsp": (50.0, 8.0), "mop": (50.0, 8.0),
        "count": 4000
    },
    {
        "plant": "eggplant", "pest": "Phomopsis Blight", "pesticide": "Zineb", "alt_pesticide": "Mancozeb",
        "temp": (27.5, 2.2), "hum": (85.0, 4.5), "moist": (72.0, 5.5),
        "ph": (6.2, 0.3), "ec": (0.7, 0.15), "p": (56.0, 9.0), "k": (245.0, 22.0), "urea": (90.0, 10.0), "tsp": (52.0, 8.0), "mop": (55.0, 8.0),
        "count": 4000
    },

    # ── 7. CARROTS ──
    {
        "plant": "carrots", "pest": "Alternaria Leaf Blight", "pesticide": "Difenoconazole", "alt_pesticide": "Mancozeb",
        "temp": (24.0, 2.2), "hum": (86.0, 4.2), "moist": (74.0, 5.0),
        "ph": (6.3, 0.28), "ec": (0.6, 0.12), "p": (48.0, 8.0), "k": (230.0, 22.0), "urea": (70.0, 10.0), "tsp": (45.0, 7.0), "mop": (55.0, 8.0),
        "count": 4000
    },
    {
        "plant": "carrots", "pest": "Carrot Rust Fly", "pesticide": "Chlorpyrifos", "alt_pesticide": "Diazinon",
        "temp": (18.5, 2.2), "hum": (72.0, 5.0), "moist": (68.0, 5.5),
        "ph": (6.5, 0.28), "ec": (0.55, 0.12), "p": (45.0, 8.0), "k": (220.0, 20.0), "urea": (65.0, 9.0), "tsp": (42.0, 7.0), "mop": (50.0, 7.0),
        "count": 4000
    },

    # ── 8. STRAWBERRIES ──
    {
        "plant": "strawberries", "pest": "Two-Spotted Spider Mite", "pesticide": "Abamectin", "alt_pesticide": "Hexythiazox",
        "temp": (32.0, 2.5), "hum": (38.0, 5.0), "moist": (34.0, 4.5),
        "ph": (6.0, 0.25), "ec": (0.65, 0.14), "p": (55.0, 9.0), "k": (260.0, 25.0), "urea": (60.0, 8.0), "tsp": (45.0, 7.0), "mop": (60.0, 8.0),
        "count": 4000
    },
    {
        "plant": "strawberries", "pest": "Gray Mold (Botrytis)", "pesticide": "Carbendazim", "alt_pesticide": "Captan",
        "temp": (19.0, 2.2), "hum": (91.0, 3.8), "moist": (82.0, 4.5),
        "ph": (5.9, 0.25), "ec": (0.6, 0.12), "p": (50.0, 8.0), "k": (240.0, 22.0), "urea": (55.0, 8.0), "tsp": (40.0, 6.0), "mop": (55.0, 7.0),
        "count": 4000
    },

    # ── 9. SUNFLOWERS ──
    {
        "plant": "sunflowers", "pest": "Head Rot / Blight", "pesticide": "Carbendazim", "alt_pesticide": "Azoxystrobin",
        "temp": (26.0, 2.2), "hum": (87.0, 4.2), "moist": (75.0, 5.2),
        "ph": (6.6, 0.3), "ec": (0.7, 0.15), "p": (52.0, 9.0), "k": (250.0, 24.0), "urea": (85.0, 11.0), "tsp": (50.0, 8.0), "mop": (58.0, 8.0),
        "count": 4000
    },
    {
        "plant": "sunflowers", "pest": "Head Borer / Semilooper", "pesticide": "Lambda-Cyhalothrin", "alt_pesticide": "Chlorpyrifos",
        "temp": (31.0, 2.5), "hum": (58.0, 5.5), "moist": (50.0, 5.5),
        "ph": (6.8, 0.3), "ec": (0.75, 0.15), "p": (58.0, 10.0), "k": (270.0, 25.0), "urea": (95.0, 12.0), "tsp": (55.0, 8.0), "mop": (62.0, 9.0),
        "count": 4000
    },

    # ── 10. CINNAMON ──
    {
        "plant": "cinnamon", "pest": "Leaf Spot / Dieback", "pesticide": "Copper Oxychloride", "alt_pesticide": "Carbendazim",
        "temp": (28.0, 2.2), "hum": (90.0, 4.0), "moist": (80.0, 5.0),
        "ph": (5.6, 0.25), "ec": (0.5, 0.12), "p": (40.0, 7.0), "k": (190.0, 20.0), "urea": (70.0, 10.0), "tsp": (38.0, 6.0), "mop": (45.0, 7.0),
        "count": 4000
    },
    {
        "plant": "cinnamon", "pest": "Shoot Borer / Caterpillars", "pesticide": "Bifenthrin", "alt_pesticide": "Lambda-Cyhalothrin",
        "temp": (30.0, 2.4), "hum": (68.0, 5.5), "moist": (58.0, 5.5),
        "ph": (5.8, 0.25), "ec": (0.55, 0.12), "p": (45.0, 8.0), "k": (205.0, 20.0), "urea": (80.0, 11.0), "tsp": (42.0, 7.0), "mop": (48.0, 7.0),
        "count": 4000
    }
]

rows = []
for p in PROFILES:
    n = p["count"]
    temps = np.clip(np.random.normal(p["temp"][0], p["temp"][1], n), 12.0, 45.0)
    hums = np.clip(np.random.normal(p["hum"][0], p["hum"][1], n), 25.0, 99.0)
    moists = np.clip(np.random.normal(p["moist"][0], p["moist"][1], n), 15.0, 98.0)
    phs = np.clip(np.random.normal(p["ph"][0], p["ph"][1], n), 4.8, 8.2)
    ecs = np.clip(np.random.normal(p["ec"][0], p["ec"][1], n), 0.15, 2.2)
    ps = np.clip(np.random.normal(p["p"][0], p["p"][1], n), 15.0, 95.0)
    ks = np.clip(np.random.normal(p["k"][0], p["k"][1], n), 110.0, 390.0)
    ureas = np.clip(np.random.normal(p["urea"][0], p["urea"][1], n), 25.0, 175.0)
    tsps = np.clip(np.random.normal(p["tsp"][0], p["tsp"][1], n), 15.0, 90.0)
    mops = np.clip(np.random.normal(p["mop"][0], p["mop"][1], n), 15.0, 90.0)

    # 93% primary pesticide, 7% alternative pesticide (realistic field practice variability)
    use_alt = np.random.rand(n) < 0.07
    pesticides = np.where(use_alt, p["alt_pesticide"], p["pesticide"])

    for i in range(n):
        rows.append({
            "pH": round(float(phs[i]), 3),
            "Soil EC": round(float(ecs[i]), 3),
            "Phosphorus": round(float(ps[i]), 2),
            "Potassium": round(float(ks[i]), 2),
            "Urea": round(float(ureas[i]), 2),
            "T.S.P": round(float(tsps[i]), 2),
            "M.O.P": round(float(mops[i]), 2),
            "Moisture": round(float(moists[i]), 2),
            "Temperature": round(float(temps[i]), 2),
            "Plant Type": p["plant"],
            "pesticide": pesticides[i],
            "pest_name": p["pest"],
            "Humidity": round(float(hums[i]), 2)
        })

df_new = pd.DataFrame(rows)
df_new = df_new.sample(frac=1.0, random_state=42).reset_index(drop=True)

out_path = "C:/Users/Om/Desktop/Smart_Pest_Surveillance/Smart_Pesticide_MultiRecommend.csv"
df_new.to_csv(out_path, index=False)

print(f"✅ Generated high-quality dataset: {df_new.shape[0]} rows, {df_new.shape[1]} columns")
print(f"📁 Saved to: {out_path}")
print(f"🌱 Crops ({df_new['Plant Type'].nunique()}): {sorted(df_new['Plant Type'].unique())}")
print(f"🐛 Distinct Pests ({df_new['pest_name'].nunique()}): {sorted(df_new['pest_name'].unique())}")
print(f"💊 Distinct Pesticides ({df_new['pesticide'].nunique()}): {sorted(df_new['pesticide'].unique())}")

