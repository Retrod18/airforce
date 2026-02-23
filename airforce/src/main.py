"""
DRDO Air Defence ML Project  
Step 3: FastAPI Backend
All endpoints for frontend team (React)
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import joblib, json, os
def to_native(obj):
    """
    Recursively convert numpy types to native Python types
    so FastAPI can serialize safely.
    """
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# INIT
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
app = FastAPI(
    title="DRDO Air Defence API",
    description="ML-powered Air Defence Classification & War Scenario Prediction",
    version="1.0.0"
)

# Allow React frontend on any port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# LOAD DATA & MODELS AT STARTUP
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
countries_df = pd.read_csv(os.path.join(DATA_DIR, "countries_profiles.csv"))
systems_df = pd.read_csv(os.path.join(DATA_DIR, "air_systems_enhanced.csv"))
scenarios_df = pd.read_csv(os.path.join(DATA_DIR, "conflict_scenarios.csv"))

# ==========================================================
# LOAD MODELS
# ==========================================================

model1 = joblib.load(os.path.join(MODEL_DIR, "model1_classifier.pkl"))
model2 = joblib.load(os.path.join(MODEL_DIR, "model2_war_outcome.pkl"))
model3 = joblib.load(os.path.join(MODEL_DIR, "model3_win_prob.pkl"))

scaler_m1 = joblib.load(os.path.join(MODEL_DIR, "scaler_m1.pkl"))
scaler_m2 = joblib.load(os.path.join(MODEL_DIR, "scaler_m2.pkl"))
sys_type_enc = joblib.load(os.path.join(MODEL_DIR, "system_type_encoder.pkl"))
with open(os.path.join(MODEL_DIR, "model_metadata.json")) as fh:
    model_meta = json.load(fh)

MODEL1_ALGO = model_meta.get("model1", {}).get("algorithm", "Classifier")
MODEL2_ALGO = model_meta.get("model2", {}).get("algorithm", "Classifier")
MODEL1_ACC = model_meta.get("model1", {}).get("accuracy", model_meta.get("model1_accuracy"))
MODEL2_ACC = model_meta.get("model2", {}).get("accuracy", model_meta.get("model2_accuracy"))

# Country coordinates for the world map
COUNTRY_COORDS = {
    "India":          {"lat": 20.59, "lng": 78.96},
    "USA":            {"lat": 37.09, "lng": -95.71},
    "Russia":         {"lat": 61.52, "lng": 105.31},
    "China":          {"lat": 35.86, "lng": 104.19},
    "Pakistan":       {"lat": 30.37, "lng": 69.34},
    "North Korea":    {"lat": 40.33, "lng": 127.51},
    "France":         {"lat": 46.22, "lng": 2.21},
    "United Kingdom": {"lat": 55.37, "lng": -3.43},
    "Israel":         {"lat": 31.04, "lng": 34.85},
    "Japan":          {"lat": 36.20, "lng": 138.25},
    "Australia":      {"lat": -25.27, "lng": 133.77},
    "Iran":           {"lat": 32.42, "lng": 53.68},
    "Turkey":         {"lat": 38.96, "lng": 35.24},
    "Saudi Arabia":   {"lat": 23.88, "lng": 45.07},
}

# Zone colour mapping (relative to India)
ZONE_COLORS = {"Red": "#ef4444", "Yellow": "#eab308", "Green": "#22c55e"}

def df_to_records(df: pd.DataFrame) -> list:
    """Convert a DataFrame to a JSON-safe list of dicts."""
    return json.loads(df.to_json(orient="records"))

def get_country_row(name: str):
    row = countries_df[countries_df["country"].str.lower() == name.lower()]
    if row.empty:
        raise HTTPException(404, f"Country '{name}' not found")
    return row.iloc[0]
def get_country_systems(country_name: str) -> pd.DataFrame:
    return systems_df[systems_df["country"] == country_name].copy()

def country_force_summary(country_name: str) -> dict:
    cs = get_country_systems(country_name)

    if cs.empty:
        # Return safe defaults instead of empty dict
        return {
            "total_systems": 0,
            "modern_count": 0,
            "traditional_count": 0,
            "modern_pct": 0,
            "avg_threat_level": 0,
            "avg_stealth": 0,
            "avg_ew": 0,
            "avg_tech_gen": 0,
            "avg_reliability": 0,
            "avg_cost_musd": 0,
            "fighter_count": 0,
            "sam_count": 0,
            "uav_count": 0,
            "helicopter_count": 0,
            "radar_count": 0,
            "missile_count": 0,
            "combat_proven_pct": 0,
        }

    return {
        "total_systems":    int(len(cs)),
        "modern_count":     int((cs["classification"] == "Modern").sum()),
        "traditional_count":int((cs["classification"] == "Traditional").sum()),
        "modern_pct":       round(float((cs["classification"] == "Modern").mean() * 100), 1),
        "avg_threat_level": round(float(cs["threat_level"].mean()), 2),
        "avg_stealth":      round(float(cs["stealth_rating"].mean()), 2),
        "avg_ew":           round(float(cs["ew_capability"].mean()), 2),
        "avg_tech_gen":     round(float(cs["tech_generation"].mean()), 2),
        "avg_reliability":  round(float(cs["reliability"].mean()), 1),
        "avg_cost_musd":    round(float(cs["cost_million_usd"].mean()), 1),
        "fighter_count":    int((cs["system_type"] == "Fighter_Aircraft").sum()),
        "sam_count":        int((cs["system_type"] == "SAM_System").sum()),
        "uav_count":        int((cs["system_type"] == "UAV_Drone").sum()),
        "helicopter_count": int((cs["system_type"] == "Helicopter").sum()),
        "radar_count":      int((cs["system_type"] == "Radar_System").sum()),
        "missile_count":    int((cs["system_type"] == "Interceptor_Missile").sum()),
        "combat_proven_pct":round(float(cs["combat_proven"].mean() * 100), 1),
    }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# PYDANTIC INPUT SCHEMAS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class SystemClassifyInput(BaseModel):
    system_name: str

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#   ROUTES
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

# â”€â”€ Health â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "online",
        "project": "DRDO Air Defence ML API",
        "version": "1.0.0",
        "endpoints": [
            "GET  /api/countries",
            "GET  /api/countries/names",
            "GET  /api/countries/{name}",
            "GET  /api/systems",
            "GET  /api/systems/names",
            "GET  /api/systems/by-name/{system_name}",
            "GET  /api/compare",
            "GET  /api/map/zones",
            "GET  /api/country/{name}/insights",
            "POST /api/predict/classify-system",
            "GET  /api/predict/war",
            "GET  /api/stats/overview",
            "GET  /api/models/info",
        ]
    }

# â”€â”€ Countries â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.get("/api/countries", tags=["Countries"])
def list_countries(risk_zone: Optional[str] = Query(None, description="Filter: Red | Yellow | Green")):
    """
    Returns all countries with basic profile.
    Optionally filter by risk_zone (Red, Yellow, Green).
    """
    df = countries_df.copy()
    if risk_zone:
        df = df[df["risk_zone"].str.lower() == risk_zone.lower()]
        if df.empty:
            raise HTTPException(404, f"No countries found for risk_zone='{risk_zone}'")

    result = []
    for _, row in df.iterrows():
        name = row["country"]
        coords = COUNTRY_COORDS.get(name, {"lat": 0, "lng": 0})
        result.append({
            "country": name,
            "iso_code": row["iso_code"],
            "risk_zone": row["risk_zone"],
            "risk_score": row["risk_score"],
            "zone_color": ZONE_COLORS[row["risk_zone"]],
            "flag_url": row["flag_url"],
            "lat": coords["lat"],
            "lng": coords["lng"],
            "gdp_billion_usd": row["gdp_billion_usd"],
            "military_budget_billion_usd": row["military_budget_billion_usd"],
            "active_personnel": row["active_personnel"],
            "combat_aircraft_count": row["combat_aircraft_count"],
            "nuclear_capable": bool(row["nuclear_capable"]),
            "relation_with_india": row["relation_with_india"],
        })
    return {"count": len(result), "countries": result}


@app.get("/api/countries/names", tags=["Countries"])
def list_country_names(
    q: Optional[str] = Query(None, description="Optional search text for autocomplete"),
    limit: int = Query(100, ge=1, le=500, description="Max results"),
):
    """
    Lightweight list for country dropdown/autocomplete.
    """
    df = countries_df.copy()
    if q:
        df = df[df["country"].str.contains(q, case=False, na=False)]
    df = df.sort_values("country").head(limit)
    cols = ["country", "iso_code", "risk_zone", "flag_url"]
    return {"count": int(len(df)), "countries": df_to_records(df[cols])}


@app.get("/api/countries/{country_name}", tags=["Countries"])
def get_country(country_name: str):
    """Full profile for a single country including systems list."""
    row = countries_df[countries_df["country"].str.lower() == country_name.lower()]
    if row.empty:
        raise HTTPException(404, f"Country '{country_name}' not found")

    row = row.iloc[0]
    coords = COUNTRY_COORDS.get(row["country"], {"lat": 0, "lng": 0})
    systems = get_country_systems(row["country"])
    force = country_force_summary(row["country"])

    return {
        **row.to_dict(),
        "lat": coords["lat"],
        "lng": coords["lng"],
        "zone_color": ZONE_COLORS[row["risk_zone"]],
        "force_summary": force,
        "systems": df_to_records(systems[[
            "system_id", "system_name", "system_type", "classification",
            "year_inducted", "threat_level", "image_url", "wikipedia_url",
            "operational_status", "combat_proven"
        ]]),
    }


# â”€â”€ Systems â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.get("/api/systems", tags=["Systems"])
def list_systems(
    country: Optional[str] = None,
    system_name: Optional[str] = Query(None, description="Search by system name (full or partial)"),
    system_type: Optional[str] = None,
    classification: Optional[str] = None,
    min_threat: Optional[float] = None,
    max_threat: Optional[float] = None,
):
    """
    All air defence systems with rich details.
    Supports filters: country, system_name, system_type, classification, threat range.
    """
    df = systems_df.copy()
    if country:
        df = df[df["country"].str.lower() == country.lower()]
    if system_name:
        df = df[df["system_name"].str.contains(system_name, case=False, na=False)]
    if system_type:
        df = df[df["system_type"].str.lower() == system_type.lower()]
    if classification:
        df = df[df["classification"].str.lower() == classification.lower()]
    if min_threat is not None:
        df = df[df["threat_level"] >= min_threat]
    if max_threat is not None:
        df = df[df["threat_level"] <= max_threat]

    if df.empty:
        raise HTTPException(404, "No systems match the given filters")

    return {"count": len(df), "systems": df_to_records(df)}


@app.get("/api/systems/names", tags=["Systems"])
def list_system_names(
    country: Optional[str] = Query(None, description="Optional country filter"),
    q: Optional[str] = Query(None, description="Optional search text for autocomplete"),
    limit: int = Query(100, ge=1, le=500, description="Max results"),
):
    """
    Lightweight list for dropdown/autocomplete by system name.
    """
    df = systems_df.copy()

    if country:
        df = df[df["country"].str.lower() == country.lower()]
    if q:
        df = df[df["system_name"].str.contains(q, case=False, na=False)]

    df = df.sort_values("system_name").head(limit)
    cols = ["system_id", "system_name", "country", "system_type", "classification", "threat_level"]

    return {"count": int(len(df)), "systems": df_to_records(df[cols])}


@app.get("/api/systems/by-name/{system_name}", tags=["Systems"])
def get_system_by_name(system_name: str):
    """
    Full specification for one system by its display name.
    Falls back to partial matching when exact name is not found.
    """
    exact = systems_df[systems_df["system_name"].str.lower() == system_name.lower()]
    if not exact.empty:
        return to_native(exact.iloc[0].to_dict())

    partial = systems_df[systems_df["system_name"].str.contains(system_name, case=False, na=False)]
    if partial.empty:
        raise HTTPException(404, f"System '{system_name}' not found")
    if len(partial) > 1:
        matches = sorted(partial["system_name"].tolist())
        raise HTTPException(
            409,
            f"Multiple systems match '{system_name}'. Please use exact name: {matches}"
        )
    return to_native(partial.iloc[0].to_dict())


# â”€â”€ Comparison (Feature 1) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.get("/api/compare", tags=["Comparison"])
def compare_countries(
    country1: str = Query(..., description="First country name"),
    country2: str = Query(..., description="Second country name"),
):
    """
    Side-by-side comparison of two countries.
    Returns all systems, stats, and chart-ready data for both.
    
    Frontend should render:
      - Radar/Spider chart of 6 key metrics
      - Bar chart per system type
      - System cards with images
    """
    def build_side(name: str):
        row = countries_df[countries_df["country"].str.lower() == name.lower()]
        if row.empty:
            raise HTTPException(404, f"Country '{name}' not found")
        row = row.iloc[0]
        force = country_force_summary(row["country"])
        systems = get_country_systems(row["country"])
        coords = COUNTRY_COORDS.get(row["country"], {"lat": 0, "lng": 0})

        # Radar chart data (normalised 0-100)
        radar = {
            "Tech Generation":  round(force["avg_tech_gen"] / 6 * 100, 1),
            "Threat Level":     round(force["avg_threat_level"] / 10 * 100, 1),
            "Stealth":          round(force["avg_stealth"] / 10 * 100, 1),
            "EW Capability":    round(force["avg_ew"] / 10 * 100, 1),
            "Reliability":      force["avg_reliability"],
            "Modernity":        force["modern_pct"],
        }

        # Systems grouped by type
        by_type = systems.groupby("system_type").apply(
            lambda g: df_to_records(g[[
                "system_id","system_name","classification","year_inducted",
                "threat_level","stealth_rating","ew_capability",
                "max_speed_kmph","range_km","reliability",
                "cost_million_usd","image_url","wikipedia_url",
                "description","combat_proven","operational_status"
            ]])
        ).to_dict()

        # Historical conflict insights
        won   = scenarios_df[(scenarios_df["attacker"] == row["country"]) & (scenarios_df["outcome"] == "Attacker_Wins")]
        lost  = scenarios_df[(scenarios_df["defender"] == row["country"]) & (scenarios_df["outcome"] == "Attacker_Wins")]
        stale = scenarios_df[
            ((scenarios_df["attacker"] == row["country"]) | (scenarios_df["defender"] == row["country"])) &
            (scenarios_df["outcome"] == "Stalemate")
        ]

        return {
            "country": row["country"],
            "iso_code": row["iso_code"],
            "flag_url": row["flag_url"],
            "risk_zone": row["risk_zone"],
            "risk_score": row["risk_score"],
            "zone_color": ZONE_COLORS[row["risk_zone"]],
            "lat": coords["lat"], "lng": coords["lng"],
            "gdp_billion_usd": row["gdp_billion_usd"],
            "military_budget_billion_usd": row["military_budget_billion_usd"],
            "active_personnel": row["active_personnel"],
            "combat_aircraft_count": row["combat_aircraft_count"],
            "nuclear_capable": bool(row["nuclear_capable"]),
            "alliance": row["alliance"],
            "key_conflicts": row["key_conflicts"],
            "relation_with_india": row["relation_with_india"],
            "force_summary": force,
            "radar_chart_data": radar,
            "systems_by_type": by_type,
            "scenario_stats": {
                "wins_as_attacker": int(len(won)),
                "losses_as_defender": int(len(lost)),
                "stalemates": int(len(stale)),
            }
        }

    side1 = build_side(country1)
    side2 = build_side(country2)

    # Direct metric comparison (for bar charts)
    metrics = list(side1["radar_chart_data"].keys())
    comparison_chart = [
        {
            "metric": m,
            country1: side1["radar_chart_data"][m],
            country2: side2["radar_chart_data"][m],
        }
        for m in metrics
    ]

    # System type counts comparison
    all_types = ["Fighter_Aircraft","SAM_System","UAV_Drone","Helicopter","Radar_System","Interceptor_Missile"]
    type_chart = [
        {
            "type": t.replace("_", " "),
            country1: side1["force_summary"].get(t.lower().replace("fighter_aircraft","fighter_count")
                        .replace("sam_system","sam_count")
                        .replace("uav_drone","uav_count")
                        .replace("helicopter","helicopter_count")
                        .replace("radar_system","radar_count")
                        .replace("interceptor_missile","missile_count"), 0),
            country2: side2["force_summary"].get(t.lower().replace("fighter_aircraft","fighter_count")
                        .replace("sam_system","sam_count")
                        .replace("uav_drone","uav_count")
                        .replace("helicopter","helicopter_count")
                        .replace("radar_system","radar_count")
                        .replace("interceptor_missile","missile_count"), 0),
        }
        for t in all_types
    ]

    return to_native({
        "country1": side1,
        "country2": side2,
        "comparison_chart": comparison_chart,
        "type_count_chart": type_chart,
    })


# â”€â”€ Map & Zones (Feature 2) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.get("/api/map/zones", tags=["War Prediction"])
def get_map_zones():
    """
    All countries with coordinates, risk zone, and colour.
    Frontend uses this to paint the interactive world map.
    """
    result = []
    for _, row in countries_df.iterrows():
        name = row["country"]
        coords = COUNTRY_COORDS.get(name, {"lat": 0, "lng": 0})
        force = country_force_summary(name)
        result.append({
            "country": name,
            "iso_code": row["iso_code"],
            "risk_zone": row["risk_zone"],
            "risk_score": int(row["risk_score"]),
            "zone_color": ZONE_COLORS[row["risk_zone"]],
            "flag_url": row["flag_url"],
            "lat": coords["lat"],
            "lng": coords["lng"],
            "nuclear_capable": bool(row["nuclear_capable"]),
            "military_budget_billion_usd": row["military_budget_billion_usd"],
            "combat_aircraft_count": int(row["combat_aircraft_count"]),
            "avg_threat_level": force.get("avg_threat_level", 0),
            "modern_pct": force.get("modern_pct", 0),
            "relation_with_india": row["relation_with_india"],
        })

    return {
        "reference_country": "India",
        "zone_legend": {
            "Red":    "Hostile / High Risk",
            "Yellow": "Neutral / Moderate Risk",
            "Green":  "Friendly / Allied / Low Risk",
        },
        "zone_counts": {z: sum(1 for r in result if r["risk_zone"] == z) for z in ["Red", "Yellow", "Green"]},
        "countries": result
    }


@app.get("/api/country/{country_name}/insights", tags=["War Prediction"])
def country_insights(country_name: str):
    """
    Detailed insights for a country when user clicks on map.
    Includes past scenario stats, top threats, strength breakdown.
    """
    row = countries_df[countries_df["country"].str.lower() == country_name.lower()]
    if row.empty:
        raise HTTPException(404, f"Country '{country_name}' not found")
    row = row.iloc[0]

    force = country_force_summary(row["country"])
    systems = get_country_systems(row["country"])

    # Top 3 most dangerous systems
    top3 = systems.nlargest(3, "threat_level")[
        ["system_name","system_type","threat_level","classification","image_url","description"]
    ]

    # Scenario history
    att_wins   = scenarios_df[(scenarios_df["attacker"]==row["country"]) & (scenarios_df["outcome"]=="Attacker_Wins")]
    def_wins   = scenarios_df[(scenarios_df["defender"]==row["country"]) & (scenarios_df["outcome"]=="Defender_Wins")]
    stalemates = scenarios_df[
        ((scenarios_df["attacker"]==row["country"]) | (scenarios_df["defender"]==row["country"])) &
        (scenarios_df["outcome"]=="Stalemate")
    ]
    total_scenarios = int(len(att_wins) + len(def_wins) + int(len(stalemates)/2))

    avg_win_prob_as_att = float(
        scenarios_df[scenarios_df["attacker"]==row["country"]]["attacker_win_probability"].mean()
    ) if len(att_wins) > 0 else 0.5

    # Strength score (0-100) for the gauge widget
    strength_score = round(
        force.get("avg_threat_level", 0) / 10 * 35 +
        force.get("modern_pct", 0) / 100 * 25 +
        force.get("avg_tech_gen", 0) / 6 * 20 +
        min(row["military_budget_billion_usd"], 300) / 300 * 20,
        1
    )

    return to_native({
        "country": row["country"],
        "flag_url": row["flag_url"],
        "iso_code": row["iso_code"],
        "risk_zone": row["risk_zone"],
        "zone_color": ZONE_COLORS[row["risk_zone"]],
        "risk_score": int(row["risk_score"]),
        "relation_with_india": row["relation_with_india"],
        "key_conflicts": row["key_conflicts"],
        "geopolitical_stance": row["geopolitical_stance"],
        "nuclear_capable": bool(row["nuclear_capable"]),
        "alliance": row["alliance"],
        "military_budget_billion_usd": row["military_budget_billion_usd"],
        "active_personnel": row["active_personnel"],
        "combat_aircraft_count": int(row["combat_aircraft_count"]),
        "force_summary": force,
        "strength_score": strength_score,
        "top_3_systems": df_to_records(top3),
        "scenario_history": {
            "wins_as_attacker": int(len(att_wins)),
            "wins_as_defender": int(len(def_wins)),
            "stalemates": int(len(stalemates)),
            "total_simulated": total_scenarios,
            "avg_win_probability_when_attacking": round(avg_win_prob_as_att, 3),
        },
        "systems_breakdown": {
            "by_classification": systems["classification"].value_counts().to_dict(),
            "by_type": systems["system_type"].value_counts().to_dict(),
        }
    })


# â”€â”€ Predictions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.post("/api/predict/classify-system", tags=["ML Predictions"])
def classify_system(data: SystemClassifyInput):
    """
    Model 1: Classify an existing system selected by name.
    Frontend should send one selected name from dropdown.
    """
    exact = systems_df[systems_df["system_name"].str.lower() == data.system_name.lower()]
    if not exact.empty:
        row = exact.iloc[0]
    else:
        partial = systems_df[systems_df["system_name"].str.contains(data.system_name, case=False, na=False)]
        if partial.empty:
            raise HTTPException(404, f"System '{data.system_name}' not found")
        if len(partial) > 1:
            matches = sorted(partial["system_name"].tolist())
            raise HTTPException(
                409,
                f"Multiple systems match '{data.system_name}'. Please use exact name: {matches}"
            )
        row = partial.iloc[0]

    type_encoded = int(sys_type_enc.transform([row["system_type"]])[0])
    features = [
        float(row["tech_generation"]), int(row["year_inducted"]), float(row["stealth_rating"]),
        float(row["ew_capability"]), float(row["max_speed_kmph"]), float(row["range_km"]),
        float(row["max_altitude_m"]), float(row["reliability"]), float(row["cost_million_usd"]),
        float(row["threat_level"]), float(row.get("payload_kg", 0.0)), type_encoded
    ]

    X = scaler_m1.transform([features])
    prediction  = model1.predict(X)[0]
    proba       = model1.predict_proba(X)[0]
    classes     = model1.classes_
    conf_dict = {c: round(float(p), 3) for c, p in zip(classes, proba)}

    return to_native({
        "selected_system": {
            "system_id": row["system_id"],
            "system_name": row["system_name"],
            "country": row["country"],
            "system_type": row["system_type"],
            "specifications": {
                "tech_generation": row["tech_generation"],
                "year_inducted": row["year_inducted"],
                "stealth_rating": row["stealth_rating"],
                "ew_capability": row["ew_capability"],
                "max_speed_kmph": row["max_speed_kmph"],
                "range_km": row["range_km"],
                "max_altitude_m": row["max_altitude_m"],
                "reliability": row["reliability"],
                "cost_million_usd": row["cost_million_usd"],
                "threat_level": row["threat_level"],
                "payload_kg": row["payload_kg"],
            },
            "dataset_classification": row["classification"],
        },
        "prediction": {
            "classification": prediction,
            "confidence": round(float(max(proba)), 3),
            "probabilities": conf_dict,
        },
        "model": MODEL1_ALGO,
        "model_accuracy": MODEL1_ACC,
    })


@app.get("/api/predict/war", tags=["ML Predictions"])
def predict_war(
    attacker_country: str = Query(..., description="Attacker country name from dropdown"),
    defender_country: str = Query(..., description="Defender country name from dropdown"),
):
    """
    Models 2 & 3: Predict war scenario outcome from selected country names.
    """
    if attacker_country.lower() == defender_country.lower():
        raise HTTPException(400, "Attacker and defender must be different countries")

    def get_country_row(name: str):
        row = countries_df[countries_df["country"].str.lower() == name.lower()]
        if row.empty:
            raise HTTPException(404, f"Country '{name}' not found")
        return row.iloc[0]

    att_row = get_country_row(attacker_country)
    dfn_row = get_country_row(defender_country)
    att_fs  = country_force_summary(att_row["country"])
    dfn_fs  = country_force_summary(dfn_row["country"])

    zone_map = {"Red": 3, "Yellow": 2, "Green": 1}

    att_budget  = float(att_row["military_budget_billion_usd"])
    dfn_budget  = float(dfn_row["military_budget_billion_usd"])
    att_ac      = float(att_row["combat_aircraft_count"])
    dfn_ac      = float(dfn_row["combat_aircraft_count"])

    threat_ratio  = att_fs["avg_threat_level"] / max(dfn_fs["avg_threat_level"], 0.01)
    tech_ratio    = att_fs["avg_tech_gen"]      / max(dfn_fs["avg_tech_gen"],    0.01)
    number_ratio  = att_ac / max(dfn_ac, 1)
    budget_ratio  = att_budget / max(dfn_budget, 0.01)

    features = [
        att_fs["avg_threat_level"], att_fs["avg_tech_gen"], att_fs["modern_pct"],
        att_fs["avg_stealth"], att_fs["avg_ew"],
        att_fs["fighter_count"], att_fs["sam_count"], att_fs["uav_count"],
        att_budget, att_ac, zone_map[att_row["risk_zone"]],

        dfn_fs["avg_threat_level"], dfn_fs["avg_tech_gen"], dfn_fs["modern_pct"],
        dfn_fs["avg_stealth"], dfn_fs["avg_ew"],
        dfn_fs["fighter_count"], dfn_fs["sam_count"], dfn_fs["uav_count"],
        dfn_budget, dfn_ac, zone_map[dfn_row["risk_zone"]],

        round(threat_ratio, 3), round(tech_ratio, 3),
        round(number_ratio, 3), round(min(budget_ratio, 100), 3),
    ]

    X = scaler_m2.transform([features])

    outcome  = model2.predict(X)[0]
    proba2   = model2.predict_proba(X)[0]
    classes2 = model2.classes_
    outcome_probs = {c: round(float(p), 3) for c, p in zip(classes2, proba2)}

    win_prob = float(np.clip(model3.predict(X)[0], 0, 1))

    att_loss = round(float(np.clip((1 - win_prob) * 40, 5, 75)), 1)
    dfn_loss = round(float(np.clip(win_prob * 40, 5, 75)), 1)
    duration = max(3, int(15 * (1 / max(abs(win_prob - 0.5) * 2, 0.05))))

    outcome_descriptions = {
        "Attacker_Wins": f"{att_row['country']} forces achieve air superiority. Significant degradation of {dfn_row['country']} air defence network.",
        "Defender_Wins": f"{dfn_row['country']} successfully repels the air campaign. {att_row['country']} forces suffer heavy attrition.",
        "Stalemate": f"Neither side achieves decisive air superiority. Prolonged attritional air war with heavy losses on both sides.",
    }

    return to_native({
        "attacker": {
            "country": att_row["country"],
            "flag_url": att_row["flag_url"],
            "risk_zone": att_row["risk_zone"],
            "zone_color": ZONE_COLORS[att_row["risk_zone"]],
            "nuclear_capable": bool(att_row["nuclear_capable"]),
            **att_fs,
        },
        "defender": {
            "country": dfn_row["country"],
            "flag_url": dfn_row["flag_url"],
            "risk_zone": dfn_row["risk_zone"],
            "zone_color": ZONE_COLORS[dfn_row["risk_zone"]],
            "nuclear_capable": bool(dfn_row["nuclear_capable"]),
            **dfn_fs,
        },
        "prediction": {
            "outcome": outcome,
            "outcome_description": outcome_descriptions[outcome],
            "attacker_win_probability": round(win_prob, 3),
            "outcome_probabilities": outcome_probs,
            "estimated_attacker_loss_pct": att_loss,
            "estimated_defender_loss_pct": dfn_loss,
            "estimated_duration_days": duration,
        },
        "advantage_factors": {
            "threat_ratio": round(threat_ratio, 3),
            "tech_ratio": round(tech_ratio, 3),
            "numbers_ratio": round(number_ratio, 3),
            "budget_ratio": round(min(budget_ratio, 100), 3),
        },
        "model": MODEL2_ALGO,
        "model_accuracy": MODEL2_ACC,
    })
# â”€â”€ Dashboard Stats â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.get("/api/stats/overview", tags=["Dashboard"])
def overview_stats():
    """Overview stats for the main dashboard / landing page."""
    return {
        "total_countries": int(len(countries_df)),
        "total_systems":   int(len(systems_df)),
        "modern_systems":  int((systems_df["classification"] == "Modern").sum()),
        "traditional_systems": int((systems_df["classification"] == "Traditional").sum()),
        "total_scenarios": int(len(scenarios_df)),
        "risk_zone_counts": countries_df["risk_zone"].value_counts().to_dict(),
        "system_type_counts": systems_df["system_type"].value_counts().to_dict(),
        "top_threat_systems": df_to_records(
            systems_df.nlargest(5, "threat_level")[
                ["system_name","country","threat_level","classification","image_url"]
            ]
        ),
        "countries_list": sorted(countries_df["country"].tolist()),
    }


# â”€â”€ Model Info â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.get("/api/models/info", tags=["Dashboard"])
def models_info():
    """Returns metadata about all trained ML models."""
    return model_meta
