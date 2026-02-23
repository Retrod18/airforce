# DRDO Air Defence API - Frontend Integration Guide

Base URL: `http://127.0.0.1:8000`
Swagger docs: `http://127.0.0.1:8000/docs`

## What Changed (Important)

The API now follows a dropdown-first workflow for user inputs.

- Users do not need to type technical specifications.
- `classify-system` now takes only `system_name`.
- `predict-war` now uses query params with country names.
- New lightweight endpoints are available for dropdown/autocomplete lists.
- Old endpoint removed: `GET /api/systems/{id}`.

## Run Backend Locally

From project root:

```bash
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
```

## Run API Auto-Tests

Keep server running in one terminal, then in another terminal:

```bash
.\.venv\Scripts\activate
python .\src\test_endpoints.py
```

If needed, ensure `BASE_URL` in `src/test_endpoints.py` is `http://127.0.0.1:8000`.

## Frontend Input Flow (Recommended)

### 1) Country dropdowns
- Call `GET /api/countries/names?q=<text>&limit=<n>`
- Use results for attacker/defender selectors and compare selectors.

### 2) System dropdown/autocomplete
- Call `GET /api/systems/names?q=<text>&country=<optional>&limit=<n>`
- Use `system_name` for system details and classify requests.

### 3) Predict/classify actions
- Classify by selected system name:
  - `POST /api/predict/classify-system`
- War prediction by selected country names:
  - `GET /api/predict/war?attacker_country=...&defender_country=...`

## Endpoint Reference

### 1) GET `/api/stats/overview`
Use for dashboard counters and top-level stats.

### 2) GET `/api/countries`
Use for full country list/cards (optional filter `risk_zone=Red|Yellow|Green`).

### 3) GET `/api/countries/names`
Use for country dropdown/autocomplete.

Query params:
- `q` (optional): search text
- `limit` (optional, default `100`, max `500`)

Example:
`GET /api/countries/names?q=in`

Response:
```json
{
  "count": 3,
  "countries": [
    {
      "country": "India",
      "iso_code": "IN",
      "risk_zone": "Green",
      "flag_url": "https://..."
    }
  ]
}
```

### 4) GET `/api/countries/{country_name}`
Full country profile + systems summary.

### 5) GET `/api/systems`
Full systems list with optional filters.

Query params:
- `country` (optional)
- `system_name` (optional, partial search)
- `system_type` (optional)
- `classification` (optional)
- `min_threat` (optional)
- `max_threat` (optional)

### 6) GET `/api/systems/names`
Use for system dropdown/autocomplete.

Query params:
- `country` (optional)
- `q` (optional)
- `limit` (optional, default `100`, max `500`)

Response fields are lightweight:
- `system_id`, `system_name`, `country`, `system_type`, `classification`, `threat_level`

### 7) GET `/api/systems/by-name/{system_name}`
Get full spec/details for a selected system name.

Behavior:
- Exact name match: returns full system object.
- If exact not found, partial match fallback is used.
- If multiple partial matches found: returns `409` with suggested exact names.

### 8) GET `/api/compare?country1=...&country2=...`
Country comparison page data:
- `country1`, `country2`
- `comparison_chart`
- `type_count_chart`

### 9) GET `/api/map/zones`
World-map data with risk zones and lat/lng.

### 10) GET `/api/country/{country_name}/insights`
Detailed country insights panel for map click.

### 11) POST `/api/predict/classify-system`
Classify by selected `system_name` only.

Request body:
```json
{
  "system_name": "Rafale (IAF)"
}
```

Response includes:
- `selected_system` (full dataset specs)
- `prediction.classification`
- `prediction.confidence`
- `prediction.probabilities`
- model metadata

Notes:
- Exact match preferred.
- Partial fallback supported (`"HQ-9B"` resolves to `"HQ-9B SAM System"` if unique).
- If ambiguous partial match, API returns `409` with exact name suggestions.

### 12) GET `/api/predict/war`
War prediction from dropdown-selected countries.

Query params (required):
- `attacker_country`
- `defender_country`

Example:
`GET /api/predict/war?attacker_country=China&defender_country=India`

Validation:
- If both countries are same, returns `400`.

Response includes:
- `attacker`, `defender`
- `prediction` (outcome, win probability, losses, duration)
- `advantage_factors`
- model metadata

### 13) GET `/api/models/info`
Model metadata for frontend info/debug panels.

## Removed Endpoint

- `GET /api/systems/{id}` is removed.
- Use `GET /api/systems/by-name/{system_name}` instead.

## Quick Frontend Checklist

- Use `/api/countries/names` for country selects.
- Use `/api/systems/names` for system selects.
- On system select, call `/api/systems/by-name/{system_name}` for details.
- For classification, send only `system_name` to `/api/predict/classify-system`.
- For war prediction, call `GET /api/predict/war` with two selected country names.
- Handle `409` from system-name lookup/classify by showing suggested exact names.

## Curl Examples

```bash
# Country dropdown
curl "http://127.0.0.1:8000/api/countries/names?q=in"

# System dropdown
curl "http://127.0.0.1:8000/api/systems/names?q=ra"

# System full details by name
curl "http://127.0.0.1:8000/api/systems/by-name/Rafale%20(IAF)"

# Classify selected system
curl -X POST "http://127.0.0.1:8000/api/predict/classify-system" \
  -H "Content-Type: application/json" \
  -d '{"system_name":"HQ-9B"}'

# War prediction with selected countries
curl "http://127.0.0.1:8000/api/predict/war?attacker_country=China&defender_country=India"
```
