"""
DRDO Air Defence ML Project
Step 1: Generate Enhanced Synthetic Dataset
- Real country names
- Wikipedia image URLs for each system
- Country profiles with geopolitical data
- Conflict scenario data for war prediction ML
"""

import pandas as pd
import numpy as np
import json, os

np.random.seed(42)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1.  COUNTRY PROFILES
#     Risk zones are from INDIA's perspective
#     Red   = Hostile / High Risk
#     Yellow = Neutral / Moderate Risk
#     Green  = Friendly / Allied / Low Risk
# ─────────────────────────────────────────────
COUNTRIES = [
    # ── RED ZONE ──────────────────────────────
    {
        "country": "China",
        "iso_code": "CN",
        "risk_zone": "Red",
        "risk_score": 85,
        "flag_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fa/Flag_of_the_People%27s_Republic_of_China.svg/320px-Flag_of_the_People%27s_Republic_of_China.svg.png",
        "gdp_billion_usd": 17700,
        "military_budget_billion_usd": 224,
        "active_personnel": 2000000,
        "combat_aircraft_count": 1571,
        "relation_with_india": "Hostile – ongoing border disputes (LAC), economic rivalry, nuclear-armed neighbour",
        "key_conflicts": "1962 Sino-Indian War, Doklam standoff 2017, Galwan clashes 2020",
        "geopolitical_stance": "Revisionist regional hegemon, BRI promoter, UNSC P5 member",
        "nuclear_capable": True,
        "alliance": "SCO, BRICS (strategic rival to India)",
    },
    {
        "country": "Pakistan",
        "iso_code": "PK",
        "risk_zone": "Red",
        "risk_score": 90,
        "flag_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Flag_of_Pakistan.svg/320px-Flag_of_Pakistan.svg.png",
        "gdp_billion_usd": 376,
        "military_budget_billion_usd": 10.4,
        "active_personnel": 654000,
        "combat_aircraft_count": 425,
        "relation_with_india": "Hostile – three full-scale wars, cross-border terrorism, Kashmir dispute",
        "key_conflicts": "1947, 1965, 1971 wars; Kargil 1999; ongoing LoC skirmishes",
        "geopolitical_stance": "Nuclear-armed; close ally of China; CPEC partner",
        "nuclear_capable": True,
        "alliance": "OIC, China-Pakistan Axis",
    },
    {
        "country": "North Korea",
        "iso_code": "KP",
        "risk_zone": "Red",
        "risk_score": 75,
        "flag_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/Flag_of_North_Korea.svg/320px-Flag_of_North_Korea.svg.png",
        "gdp_billion_usd": 18,
        "military_budget_billion_usd": 4,
        "active_personnel": 1280000,
        "combat_aircraft_count": 563,
        "relation_with_india": "Low direct risk; indirect threat via missile proliferation to adversaries",
        "key_conflicts": "Korean War 1950-53; constant military provocations",
        "geopolitical_stance": "Hermit kingdom; nuclear-armed; missile proliferator",
        "nuclear_capable": True,
        "alliance": "Isolated; tacit China support",
    },
    # ── YELLOW ZONE ───────────────────────────
    {
        "country": "Russia",
        "iso_code": "RU",
        "risk_zone": "Yellow",
        "risk_score": 45,
        "flag_url": "https://upload.wikimedia.org/wikipedia/en/thumb/f/f3/Flag_of_Russia.svg/320px-Flag_of_Russia.svg.png",
        "gdp_billion_usd": 1900,
        "military_budget_billion_usd": 86,
        "active_personnel": 900000,
        "combat_aircraft_count": 1172,
        "relation_with_india": "Complex – major arms supplier, historical friend, but rapprochement with China and Pakistan concerns India",
        "key_conflicts": "Ukraine War 2022-; Syria; Georgia 2008",
        "geopolitical_stance": "Great power; UNSC P5; major arms exporter",
        "nuclear_capable": True,
        "alliance": "CSTO; SCO; BRICS",
    },
    {
        "country": "Iran",
        "iso_code": "IR",
        "risk_zone": "Yellow",
        "risk_score": 50,
        "flag_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Flag_of_Iran.svg/320px-Flag_of_Iran.svg.png",
        "gdp_billion_usd": 367,
        "military_budget_billion_usd": 9.9,
        "active_personnel": 525000,
        "combat_aircraft_count": 337,
        "relation_with_india": "Neutral-positive; Chabahar port partner; energy ties but concerns about proxy networks",
        "key_conflicts": "Iran-Iraq War 1980-88; proxy conflicts in Middle East",
        "geopolitical_stance": "Theocratic regional power; proxy warfare doctrine; near-nuclear",
        "nuclear_capable": False,
        "alliance": "SCO observer; Axis of Resistance",
    },
    {
        "country": "Turkey",
        "iso_code": "TR",
        "risk_zone": "Yellow",
        "risk_score": 40,
        "flag_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Flag_of_Turkey.svg/320px-Flag_of_Turkey.svg.png",
        "gdp_billion_usd": 1154,
        "military_budget_billion_usd": 18.5,
        "active_personnel": 355000,
        "combat_aircraft_count": 207,
        "relation_with_india": "Neutral; supports Pakistan on Kashmir; growing economic ties with India",
        "key_conflicts": "Syria; Libya; Nagorno-Karabakh (support); Cyprus",
        "geopolitical_stance": "NATO member but increasingly independent; major drone exporter",
        "nuclear_capable": False,
        "alliance": "NATO (strained); OIC",
    },
    {
        "country": "Saudi Arabia",
        "iso_code": "SA",
        "risk_zone": "Yellow",
        "risk_score": 30,
        "flag_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Flag_of_Saudi_Arabia.svg/320px-Flag_of_Saudi_Arabia.svg.png",
        "gdp_billion_usd": 1108,
        "military_budget_billion_usd": 75,
        "active_personnel": 257000,
        "combat_aircraft_count": 234,
        "relation_with_india": "Positive economic ties; large Indian diaspora; growing defence engagement",
        "key_conflicts": "Yemen War 2015-; Gulf War 1991",
        "geopolitical_stance": "US-aligned; oil power; normalization with Israel",
        "nuclear_capable": False,
        "alliance": "GCC; Arab League; US strategic partner",
    },
    # ── GREEN ZONE ────────────────────────────
    {
        "country": "India",
        "iso_code": "IN",
        "risk_zone": "Green",
        "risk_score": 0,
        "flag_url": "https://upload.wikimedia.org/wikipedia/en/thumb/4/41/Flag_of_India.svg/320px-Flag_of_India.svg.png",
        "gdp_billion_usd": 3730,
        "military_budget_billion_usd": 81.4,
        "active_personnel": 1460000,
        "combat_aircraft_count": 630,
        "relation_with_india": "Home country – reference nation",
        "key_conflicts": "1947, 1965, 1971 wars with Pakistan; 1962 with China; Kargil 1999",
        "geopolitical_stance": "Strategic autonomy; world's largest democracy; QUAD member",
        "nuclear_capable": True,
        "alliance": "QUAD; SCO; BRICS; Non-Aligned",
    },
    {
        "country": "USA",
        "iso_code": "US",
        "risk_zone": "Green",
        "risk_score": 5,
        "flag_url": "https://upload.wikimedia.org/wikipedia/en/thumb/a/a4/Flag_of_the_United_States.svg/320px-Flag_of_the_United_States.svg.png",
        "gdp_billion_usd": 27360,
        "military_budget_billion_usd": 886,
        "active_personnel": 1400000,
        "combat_aircraft_count": 2085,
        "relation_with_india": "Strategic partner – QUAD ally, major defence supplier (P-8I, C-17, Apache), tech cooperation",
        "key_conflicts": "Iraq, Afghanistan, Gulf War; proxy conflicts globally",
        "geopolitical_stance": "Global hegemon; NATO leader; Indo-Pacific pivot",
        "nuclear_capable": True,
        "alliance": "NATO; AUKUS; QUAD",
    },
    {
        "country": "France",
        "iso_code": "FR",
        "risk_zone": "Green",
        "risk_score": 5,
        "flag_url": "https://upload.wikimedia.org/wikipedia/en/thumb/c/c3/Flag_of_France.svg/320px-Flag_of_France.svg.png",
        "gdp_billion_usd": 3050,
        "military_budget_billion_usd": 59,
        "active_personnel": 205000,
        "combat_aircraft_count": 254,
        "relation_with_india": "Privileged strategic partner – Rafale deal, submarine cooperation, IOR alignment",
        "key_conflicts": "Mali; Libya; Gulf War",
        "geopolitical_stance": "EU/NATO leader; independent nuclear deterrent; active Indo-Pacific presence",
        "nuclear_capable": True,
        "alliance": "NATO; EU",
    },
    {
        "country": "United Kingdom",
        "iso_code": "GB",
        "risk_zone": "Green",
        "risk_score": 5,
        "flag_url": "https://upload.wikimedia.org/wikipedia/en/thumb/a/ae/Flag_of_the_United_Kingdom.svg/320px-Flag_of_the_United_Kingdom.svg.png",
        "gdp_billion_usd": 3090,
        "military_budget_billion_usd": 68,
        "active_personnel": 150000,
        "combat_aircraft_count": 152,
        "relation_with_india": "Positive – JETCO FTA negotiations, defence tech tie-ups, strong diaspora links",
        "key_conflicts": "Iraq; Afghanistan; Falklands; Gulf War",
        "geopolitical_stance": "Post-Brexit global Britain; AUKUS member; P5 UNSC",
        "nuclear_capable": True,
        "alliance": "NATO; AUKUS; Five Eyes",
    },
    {
        "country": "Israel",
        "iso_code": "IL",
        "risk_zone": "Green",
        "risk_score": 8,
        "flag_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Flag_of_Israel.svg/320px-Flag_of_Israel.svg.png",
        "gdp_billion_usd": 528,
        "military_budget_billion_usd": 23.5,
        "active_personnel": 170000,
        "combat_aircraft_count": 241,
        "relation_with_india": "Strong partner – biggest importer of Israeli defence tech (Barak-8, Heron, Spike missiles)",
        "key_conflicts": "Arab-Israeli Wars; Gaza; Lebanon; Iran proxy conflicts",
        "geopolitical_stance": "Regional nuclear power (undeclared); world's best iron-dome layered air defence",
        "nuclear_capable": True,
        "alliance": "US-aligned; Abraham Accords; India partnership",
    },
    {
        "country": "Japan",
        "iso_code": "JP",
        "risk_zone": "Green",
        "risk_score": 5,
        "flag_url": "https://upload.wikimedia.org/wikipedia/en/thumb/9/9e/Flag_of_Japan.svg/320px-Flag_of_Japan.svg.png",
        "gdp_billion_usd": 4200,
        "military_budget_billion_usd": 51,
        "active_personnel": 247000,
        "combat_aircraft_count": 363,
        "relation_with_india": "Positive – Japan-India Special Strategic and Global Partnership; QUAD ally",
        "key_conflicts": "WWII; no active conflicts post-1945",
        "geopolitical_stance": "Pacifist constitution (evolving); US treaty ally; major naval power",
        "nuclear_capable": False,
        "alliance": "US-Japan Treaty; QUAD",
    },
    {
        "country": "Australia",
        "iso_code": "AU",
        "risk_zone": "Green",
        "risk_score": 5,
        "flag_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Flag_of_Australia.svg/320px-Flag_of_Australia.svg.png",
        "gdp_billion_usd": 1720,
        "military_budget_billion_usd": 32,
        "active_personnel": 60000,
        "combat_aircraft_count": 106,
        "relation_with_india": "Positive – Comprehensive Strategic Partnership; QUAD ally; Malabar exercise participant",
        "key_conflicts": "Iraq; Afghanistan",
        "geopolitical_stance": "AUKUS nuclear submarine partner; Five Eyes; Indo-Pacific focus",
        "nuclear_capable": False,
        "alliance": "AUKUS; QUAD; Five Eyes; US alliance",
    },
]

# ─────────────────────────────────────────────
# 2.  AIR DEFENCE SYSTEMS PER COUNTRY
#     Each entry has: name, type, classification,
#     full specs, image_url, wikipedia_url
# ─────────────────────────────────────────────
# Classification:  Traditional = pre-2005 design/generation
#                  Modern      = 2005-onwards design/generation

SYSTEMS = [
    # ═══════ INDIA ═══════
    {
        "country": "India", "system_name": "Su-30MKI Flanker-H",
        "system_type": "Fighter_Aircraft", "classification": "Modern",
        "year_inducted": 2002, "tech_generation": 4.5,
        "max_speed_kmph": 2120, "range_km": 3000, "max_altitude_m": 17300,
        "stealth_rating": 3.5, "ew_capability": 7.2, "payload_kg": 8000,
        "reliability": 82, "cost_million_usd": 62, "threat_level": 7.8,
        "operational_status": "Active", "combat_proven": True,
        "description": "India's primary air superiority fighter. Twin-engine supermaneuverable multirole aircraft with thrust-vectoring. Forms the backbone of IAF. Over 260 in service.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/Su-30MKI_Aero_India_2013_01.jpg/320px-Su-30MKI_Aero_India_2013_01.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Sukhoi_Su-30MKI",
    },
    {
        "country": "India", "system_name": "HAL Tejas Mk1A",
        "system_type": "Fighter_Aircraft", "classification": "Modern",
        "year_inducted": 2016, "tech_generation": 4.5,
        "max_speed_kmph": 1350, "range_km": 1850, "max_altitude_m": 15240,
        "stealth_rating": 3.0, "ew_capability": 7.5, "payload_kg": 4000,
        "reliability": 75, "cost_million_usd": 78, "threat_level": 7.0,
        "operational_status": "Active", "combat_proven": False,
        "description": "India's indigenous light combat aircraft. Mk1A features AESA radar, advanced EW suite, AAR probe. IAF ordered 83 Mk1A in 2021.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Tejas_Mk1A.jpg/320px-Tejas_Mk1A.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/HAL_Tejas",
    },
    {
        "country": "India", "system_name": "Rafale (IAF)",
        "system_type": "Fighter_Aircraft", "classification": "Modern",
        "year_inducted": 2020, "tech_generation": 4.5,
        "max_speed_kmph": 1915, "range_km": 3700, "max_altitude_m": 15240,
        "stealth_rating": 4.5, "ew_capability": 9.0, "payload_kg": 9500,
        "reliability": 92, "cost_million_usd": 218, "threat_level": 9.0,
        "operational_status": "Active", "combat_proven": True,
        "description": "France's omnirole 4.5-gen fighter. India operates 36 Rafale with India-specific enhancements (SPECTRA EW, Meteor BVR missile, MICA). Based at Ambala & Hasimara.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/Rafale_Farnborough_Airshow_2010.jpg/320px-Rafale_Farnborough_Airshow_2010.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Dassault_Rafale",
    },
    {
        "country": "India", "system_name": "MiG-29UPG",
        "system_type": "Fighter_Aircraft", "classification": "Traditional",
        "year_inducted": 1987, "tech_generation": 3.5,
        "max_speed_kmph": 2445, "range_km": 1430, "max_altitude_m": 18000,
        "stealth_rating": 2.0, "ew_capability": 5.0, "payload_kg": 4500,
        "reliability": 68, "cost_million_usd": 38, "threat_level": 5.5,
        "operational_status": "Active", "combat_proven": True,
        "description": "Legacy fighter upgraded with glass cockpit, R-77 missiles and new radar. IAF operates ~60. Primarily used for air defence interception role.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/da/Mikoyan-Gurevich_MiG-29_%28edit1%29.jpg/320px-Mikoyan-Gurevich_MiG-29_%28edit1%29.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Mikoyan_MiG-29",
    },
    {
        "country": "India", "system_name": "LCH Prachand",
        "system_type": "Helicopter", "classification": "Modern",
        "year_inducted": 2022, "tech_generation": 4.5,
        "max_speed_kmph": 330, "range_km": 700, "max_altitude_m": 6500,
        "stealth_rating": 4.0, "ew_capability": 6.5, "payload_kg": 700,
        "reliability": 80, "cost_million_usd": 33, "threat_level": 7.5,
        "operational_status": "Active", "combat_proven": False,
        "description": "India's indigenous light combat helicopter. Designed for high-altitude warfare in Himalayas. Armed with 20mm gun, rockets, MISTRAL air-to-air missiles.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/LCH_HAL.jpg/320px-LCH_HAL.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/HAL_Light_Combat_Helicopter",
    },
    {
        "country": "India", "system_name": "S-400 Triumf (India)",
        "system_type": "SAM_System", "classification": "Modern",
        "year_inducted": 2021, "tech_generation": 5.5,
        "max_speed_kmph": 17000, "range_km": 400, "max_altitude_m": 30000,
        "stealth_rating": 0, "ew_capability": 9.5, "payload_kg": 0,
        "reliability": 95, "cost_million_usd": 860, "threat_level": 9.8,
        "operational_status": "Active", "combat_proven": True,
        "description": "World's most advanced long-range SAM system. India operates first squadron in Punjab. Can simultaneously track 100 targets and engage 36. Acquired despite US CAATSA pressure.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/S-400_Triumf_by_Vitaly_Kuzmin.jpg/320px-S-400_Triumf_by_Vitaly_Kuzmin.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/S-400_missile_system",
    },
    {
        "country": "India", "system_name": "Akash Mk2 SAM",
        "system_type": "SAM_System", "classification": "Modern",
        "year_inducted": 2014, "tech_generation": 4.0,
        "max_speed_kmph": 6000, "range_km": 40, "max_altitude_m": 18000,
        "stealth_rating": 0, "ew_capability": 6.0, "payload_kg": 0,
        "reliability": 80, "cost_million_usd": 65, "threat_level": 7.2,
        "operational_status": "Active", "combat_proven": False,
        "description": "India's indigenous medium-range SAM. Fully mobile system with phased-array radar. Can engage 4 targets simultaneously. DRDO-developed. Major export success with Armenia.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6b/Akash_Missile_Systems.jpg/320px-Akash_Missile_Systems.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Akash_(missile)",
    },
    {
        "country": "India", "system_name": "Rustom-II (MALE UAV)",
        "system_type": "UAV_Drone", "classification": "Modern",
        "year_inducted": 2024, "tech_generation": 4.5,
        "max_speed_kmph": 225, "range_km": 1000, "max_altitude_m": 10500,
        "stealth_rating": 3.0, "ew_capability": 5.0, "payload_kg": 350,
        "reliability": 72, "cost_million_usd": 9.5, "threat_level": 6.0,
        "operational_status": "Active", "combat_proven": False,
        "description": "India's medium-altitude long-endurance drone developed by ADE/DRDO. Carries SAR radar, EO/IR sensors. Endurance of 24 hours. Designed to replace Heron UAVs.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Rustom_II_MALE_UAV.jpg/320px-Rustom_II_MALE_UAV.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/DRDO_Rustom",
    },
    {
        "country": "India", "system_name": "ARUDHRA Radar",
        "system_type": "Radar_System", "classification": "Modern",
        "year_inducted": 2015, "tech_generation": 4.5,
        "max_speed_kmph": 0, "range_km": 450, "max_altitude_m": 0,
        "stealth_rating": 0, "ew_capability": 8.0, "payload_kg": 0,
        "reliability": 88, "cost_million_usd": 28, "threat_level": 5.0,
        "operational_status": "Active", "combat_proven": False,
        "description": "DRDO-developed medium-power radar for air traffic management and defence. Can track 1500 targets simultaneously. Deployed across multiple IAF bases.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/ARUDHRA_Radar_IAF.jpg/320px-ARUDHRA_Radar_IAF.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/DRDO_Arudhra",
    },
    {
        "country": "India", "system_name": "Mirage 2000H",
        "system_type": "Fighter_Aircraft", "classification": "Traditional",
        "year_inducted": 1985, "tech_generation": 3.5,
        "max_speed_kmph": 2340, "range_km": 1850, "max_altitude_m": 17060,
        "stealth_rating": 2.5, "ew_capability": 5.5, "payload_kg": 6300,
        "reliability": 70, "cost_million_usd": 42, "threat_level": 6.0,
        "operational_status": "Active", "combat_proven": True,
        "description": "Battle-proven IAF workhorse. Participated in Kargil War (1999) – precision strikes. Underwent upgrade program. Still in service, being replaced by Rafale.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/Mirage_2000_img_3924.jpg/320px-Mirage_2000_img_3924.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Dassault_Mirage_2000",
    },

    # ═══════ USA ═══════
    {
        "country": "USA", "system_name": "F-22 Raptor",
        "system_type": "Fighter_Aircraft", "classification": "Modern",
        "year_inducted": 2005, "tech_generation": 5.0,
        "max_speed_kmph": 2414, "range_km": 2960, "max_altitude_m": 19800,
        "stealth_rating": 9.5, "ew_capability": 9.5, "payload_kg": 4500,
        "reliability": 85, "cost_million_usd": 334, "threat_level": 9.8,
        "operational_status": "Active", "combat_proven": True,
        "description": "World's first operational 5th-gen air superiority fighter. All-aspect stealth, supercruise capability. 186 in USAF service. Unmatched BVR combat capability.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/F-22_Raptor_edited.jpg/320px-F-22_Raptor_edited.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Lockheed_Martin_F-22_Raptor",
    },
    {
        "country": "USA", "system_name": "F-35A Lightning II",
        "system_type": "Fighter_Aircraft", "classification": "Modern",
        "year_inducted": 2015, "tech_generation": 5.0,
        "max_speed_kmph": 1960, "range_km": 2220, "max_altitude_m": 15240,
        "stealth_rating": 9.0, "ew_capability": 9.8, "payload_kg": 8160,
        "reliability": 88, "cost_million_usd": 110, "threat_level": 9.5,
        "operational_status": "Active", "combat_proven": True,
        "description": "Multi-role 5th-gen stealth strike fighter. Most advanced sensor fusion ever built. 800+ delivered globally. AN/APG-81 AESA radar, DAS system, EOTS.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/F-35A_flight_%28cropped%29.jpg/320px-F-35A_flight_%28cropped%29.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Lockheed_Martin_F-35_Lightning_II",
    },
    {
        "country": "USA", "system_name": "F-15EX Eagle II",
        "system_type": "Fighter_Aircraft", "classification": "Modern",
        "year_inducted": 2021, "tech_generation": 4.5,
        "max_speed_kmph": 3017, "range_km": 1967, "max_altitude_m": 20000,
        "stealth_rating": 3.5, "ew_capability": 9.0, "payload_kg": 13381,
        "reliability": 93, "cost_million_usd": 87, "threat_level": 9.2,
        "operational_status": "Active", "combat_proven": False,
        "description": "Latest evolution of the legendary F-15. Largest payload of any fighter (22 AIM-120 missiles). Fly-by-wire, EPAWSS EW suite. Never lost in air-to-air combat lineage.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/eb/F-15EX_first_flight.jpg/320px-F-15EX_first_flight.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Boeing_F-15EX_Eagle_II",
    },
    {
        "country": "USA", "system_name": "MQ-9 Reaper",
        "system_type": "UAV_Drone", "classification": "Modern",
        "year_inducted": 2007, "tech_generation": 4.5,
        "max_speed_kmph": 482, "range_km": 1852, "max_altitude_m": 15240,
        "stealth_rating": 2.5, "ew_capability": 6.0, "payload_kg": 1700,
        "reliability": 90, "cost_million_usd": 32, "threat_level": 8.0,
        "operational_status": "Active", "combat_proven": True,
        "description": "Hunter-killer MALE drone. Can loiter 27 hours. Armed with Hellfire missiles, GBU-12 bombs. 300+ kills of HVTs. Operated by USAF, CIA, allies in 6 countries.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/57/MQ-9_Reaper_in_flight_%28edit%29.jpg/320px-MQ-9_Reaper_in_flight_%28edit%29.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/General_Atomics_MQ-9_Reaper",
    },
    {
        "country": "USA", "system_name": "THAAD System",
        "system_type": "SAM_System", "classification": "Modern",
        "year_inducted": 2008, "tech_generation": 5.5,
        "max_speed_kmph": 10800, "range_km": 200, "max_altitude_m": 150000,
        "stealth_rating": 0, "ew_capability": 9.0, "payload_kg": 0,
        "reliability": 97, "cost_million_usd": 3000, "threat_level": 9.5,
        "operational_status": "Active", "combat_proven": True,
        "description": "Terminal High Altitude Area Defense. Hit-to-kill technology for ballistic missiles. Deployed in South Korea, Guam, UAE, Israel. AN/TPY-2 radar, 100% intercept rate in tests.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/THAAD_battery.jpg/320px-THAAD_battery.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Terminal_High_Altitude_Area_Defense",
    },
    {
        "country": "USA", "system_name": "Patriot PAC-3 MSE",
        "system_type": "SAM_System", "classification": "Modern",
        "year_inducted": 2016, "tech_generation": 5.0,
        "max_speed_kmph": 8100, "range_km": 35, "max_altitude_m": 24000,
        "stealth_rating": 0, "ew_capability": 8.5, "payload_kg": 0,
        "reliability": 93, "cost_million_usd": 4, "threat_level": 8.8,
        "operational_status": "Active", "combat_proven": True,
        "description": "Latest Patriot interceptor with extended range. Hit-to-kill kinetic warhead. Combat proven in Ukraine. Used by 18 countries. Proven against TBMs and cruise missiles.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/Patriot_missile_launch_b.jpg/320px-Patriot_missile_launch_b.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Patriot_missile_system",
    },
    {
        "country": "USA", "system_name": "AH-64E Apache Guardian",
        "system_type": "Helicopter", "classification": "Modern",
        "year_inducted": 2011, "tech_generation": 4.5,
        "max_speed_kmph": 293, "range_km": 476, "max_altitude_m": 4573,
        "stealth_rating": 2.5, "ew_capability": 7.0, "payload_kg": 1701,
        "reliability": 87, "cost_million_usd": 35, "threat_level": 8.5,
        "operational_status": "Active", "combat_proven": True,
        "description": "World's premier attack helicopter. Fire Control Radar, AGM-114 Hellfire, Stinger AAM. India operates 22 Apaches. Proven in Iraq, Afghanistan, Gulf War.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/AH-64_Apache.jpg/320px-AH-64_Apache.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Boeing_AH-64_Apache",
    },
    {
        "country": "USA", "system_name": "F-16 Block 70/72",
        "system_type": "Fighter_Aircraft", "classification": "Modern",
        "year_inducted": 2023, "tech_generation": 4.5,
        "max_speed_kmph": 2120, "range_km": 1600, "max_altitude_m": 15240,
        "stealth_rating": 3.0, "ew_capability": 8.0, "payload_kg": 7700,
        "reliability": 90, "cost_million_usd": 65, "threat_level": 8.0,
        "operational_status": "Active", "combat_proven": True,
        "description": "Most advanced F-16 variant. AN/APG-83 SABR AESA radar, upgraded avionics. New production aircraft offered to India. Combat proven in 25 countries.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/F-16_June_2008.jpg/320px-F-16_June_2008.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/General_Dynamics_F-16_Fighting_Falcon",
    },

    # ═══════ RUSSIA ═══════
    {
        "country": "Russia", "system_name": "Su-57 Felon",
        "system_type": "Fighter_Aircraft", "classification": "Modern",
        "year_inducted": 2020, "tech_generation": 5.0,
        "max_speed_kmph": 2600, "range_km": 3500, "max_altitude_m": 20000,
        "stealth_rating": 7.5, "ew_capability": 9.5, "payload_kg": 10000,
        "reliability": 78, "cost_million_usd": 50, "threat_level": 9.2,
        "operational_status": "Active", "combat_proven": True,
        "description": "Russia's 5th-gen stealth fighter. Internal weapons bays, supercruise. Used in Ukraine conflict. Features Sh121 EW suite – most powerful fighter-EW in world.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Su-57_in_2019.jpg/320px-Su-57_in_2019.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Sukhoi_Su-57",
    },
    {
        "country": "Russia", "system_name": "Su-35S Flanker-E",
        "system_type": "Fighter_Aircraft", "classification": "Modern",
        "year_inducted": 2014, "tech_generation": 4.5,
        "max_speed_kmph": 2500, "range_km": 3600, "max_altitude_m": 18000,
        "stealth_rating": 3.5, "ew_capability": 8.5, "payload_kg": 8000,
        "reliability": 82, "cost_million_usd": 65, "threat_level": 8.8,
        "operational_status": "Active", "combat_proven": True,
        "description": "4++ generation super-maneuverable fighter. Irbis-E radar (400km range), 3D TVC engines. ~100 in VKS service. Used extensively in Syria and Ukraine.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Su-35S_in_2015.jpg/320px-Su-35S_in_2015.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Sukhoi_Su-35",
    },
    {
        "country": "Russia", "system_name": "S-500 Prometheus",
        "system_type": "SAM_System", "classification": "Modern",
        "year_inducted": 2021, "tech_generation": 6.0,
        "max_speed_kmph": 25200, "range_km": 600, "max_altitude_m": 200000,
        "stealth_rating": 0, "ew_capability": 9.8, "payload_kg": 0,
        "reliability": 90, "cost_million_usd": 1200, "threat_level": 10.0,
        "operational_status": "Active", "combat_proven": False,
        "description": "World's most advanced SAM system. Can intercept hypersonic glide vehicles, satellites, ICBM warheads. 200km altitude engagement. Fully networked with S-400.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/S-500_Prometheus_missile_system.jpg/320px-S-500_Prometheus_missile_system.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/S-500_missile_system",
    },
    {
        "country": "Russia", "system_name": "S-400 Triumf (Russia)",
        "system_type": "SAM_System", "classification": "Modern",
        "year_inducted": 2007, "tech_generation": 5.5,
        "max_speed_kmph": 17000, "range_km": 400, "max_altitude_m": 30000,
        "stealth_rating": 0, "ew_capability": 9.5, "payload_kg": 0,
        "reliability": 95, "cost_million_usd": 500, "threat_level": 9.8,
        "operational_status": "Active", "combat_proven": True,
        "description": "Proven long-range SAM. Russia operates 40+ batteries. Sold to India, China, Turkey. Can engage stealth aircraft. 91N6E Big Bird radar detects F-35.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/S-400_Triumf_by_Vitaly_Kuzmin.jpg/320px-S-400_Triumf_by_Vitaly_Kuzmin.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/S-400_missile_system",
    },
    {
        "country": "Russia", "system_name": "Ka-52M Alligator",
        "system_type": "Helicopter", "classification": "Modern",
        "year_inducted": 2020, "tech_generation": 4.5,
        "max_speed_kmph": 300, "range_km": 460, "max_altitude_m": 5500,
        "stealth_rating": 3.0, "ew_capability": 8.0, "payload_kg": 2000,
        "reliability": 80, "cost_million_usd": 18, "threat_level": 8.2,
        "operational_status": "Active", "combat_proven": True,
        "description": "Side-by-side twin-seat attack helicopter. Advanced Arbalet MMW radar, Hermes-A anti-tank missiles. Widely used in Ukraine. Export version Ka-52K for aircraft carriers.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Ka-52_%22Alligator%22_MAKS2013.jpg/320px-Ka-52_%22Alligator%22_MAKS2013.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Kamov_Ka-52",
    },
    {
        "country": "Russia", "system_name": "S-70 Okhotnik-B (UCAV)",
        "system_type": "UAV_Drone", "classification": "Modern",
        "year_inducted": 2023, "tech_generation": 5.5,
        "max_speed_kmph": 1000, "range_km": 6000, "max_altitude_m": 18000,
        "stealth_rating": 8.0, "ew_capability": 7.0, "payload_kg": 6000,
        "reliability": 72, "cost_million_usd": 40, "threat_level": 9.0,
        "operational_status": "Active", "combat_proven": False,
        "description": "Heavy stealth combat drone (20-tonne UCAV). Designed to fly with Su-57 as loyal wingman. Subsonic flying wing design. Carries air-launched cruise missiles.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/S-70_Okhotnik_UCAV.jpg/320px-S-70_Okhotnik_UCAV.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Sukhoi_S-70_Okhotnik-B",
    },

    # ═══════ CHINA ═══════
    {
        "country": "China", "system_name": "J-20 Mighty Dragon",
        "system_type": "Fighter_Aircraft", "classification": "Modern",
        "year_inducted": 2017, "tech_generation": 5.0,
        "max_speed_kmph": 2100, "range_km": 2000, "max_altitude_m": 20000,
        "stealth_rating": 7.5, "ew_capability": 8.5, "payload_kg": 11000,
        "reliability": 80, "cost_million_usd": 100, "threat_level": 9.0,
        "operational_status": "Active", "combat_proven": False,
        "description": "China's 5th-gen air superiority fighter. Features canard-delta design, AESA radar, TY-90 IR missile. ~150 in PLAAF service. Primary counter to F-22/F-35.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/J-20_2018.jpg/320px-J-20_2018.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Chengdu_J-20",
    },
    {
        "country": "China", "system_name": "J-16 Flanker Derivative",
        "system_type": "Fighter_Aircraft", "classification": "Modern",
        "year_inducted": 2013, "tech_generation": 4.5,
        "max_speed_kmph": 2200, "range_km": 4000, "max_altitude_m": 18000,
        "stealth_rating": 3.0, "ew_capability": 8.0, "payload_kg": 12000,
        "reliability": 82, "cost_million_usd": 70, "threat_level": 8.5,
        "operational_status": "Active", "combat_proven": False,
        "description": "China's premier multirole strike fighter derived from Su-30. AESA radar, 12 hardpoints. ~200 in PLAAF. J-16D variant is dedicated electronic warfare aircraft.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/90/J-16_in_2018.jpg/320px-J-16_in_2018.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Shenyang_J-16",
    },
    {
        "country": "China", "system_name": "J-10C Vigorous Dragon",
        "system_type": "Fighter_Aircraft", "classification": "Modern",
        "year_inducted": 2018, "tech_generation": 4.5,
        "max_speed_kmph": 2100, "range_km": 1850, "max_altitude_m": 18000,
        "stealth_rating": 3.5, "ew_capability": 7.5, "payload_kg": 7200,
        "reliability": 83, "cost_million_usd": 49, "threat_level": 8.0,
        "operational_status": "Active", "combat_proven": False,
        "description": "China's AESA-equipped 4.5-gen lightweight fighter. PL-15 and PL-10 missiles. Pakistan operates export J-10CE. Direct competitor to India's Tejas Mk2.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/J-10C_of_the_PLAAF.jpg/320px-J-10C_of_the_PLAAF.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Chengdu_J-10",
    },
    {
        "country": "China", "system_name": "HQ-9B SAM System",
        "system_type": "SAM_System", "classification": "Modern",
        "year_inducted": 2016, "tech_generation": 5.0,
        "max_speed_kmph": 14000, "range_km": 260, "max_altitude_m": 27000,
        "stealth_rating": 0, "ew_capability": 8.5, "payload_kg": 0,
        "reliability": 88, "cost_million_usd": 500, "threat_level": 9.0,
        "operational_status": "Active", "combat_proven": False,
        "description": "China's premier long-range SAM inspired by S-300. Phased-array radar, 260km range. Deployed along LAC near Pangong Tso. China exports to multiple countries.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/HQ-9_SAM_system.jpg/320px-HQ-9_SAM_system.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/HQ-9",
    },
    {
        "country": "China", "system_name": "Wing Loong II (MALE UAV)",
        "system_type": "UAV_Drone", "classification": "Modern",
        "year_inducted": 2017, "tech_generation": 4.5,
        "max_speed_kmph": 370, "range_km": 2000, "max_altitude_m": 9000,
        "stealth_rating": 2.0, "ew_capability": 5.5, "payload_kg": 480,
        "reliability": 80, "cost_million_usd": 4.5, "threat_level": 6.5,
        "operational_status": "Active", "combat_proven": True,
        "description": "CASC combat drone widely exported. Carries CM-102 ARMs, LS-6 bombs, Blue Arrow anti-tank missiles. Used in Libya, Ethiopia, UAE. China's answer to MQ-9.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Wing_Loong_II_UAV.jpg/320px-Wing_Loong_II_UAV.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/CASC_Wing_Loong",
    },
    {
        "country": "China", "system_name": "Z-10ME Attack Helicopter",
        "system_type": "Helicopter", "classification": "Modern",
        "year_inducted": 2003, "tech_generation": 4.0,
        "max_speed_kmph": 270, "range_km": 800, "max_altitude_m": 6400,
        "stealth_rating": 2.5, "ew_capability": 6.0, "payload_kg": 1500,
        "reliability": 78, "cost_million_usd": 15, "threat_level": 7.0,
        "operational_status": "Active", "combat_proven": False,
        "description": "China's dedicated attack helicopter. Features millimeter-wave radar, HJ-10 anti-tank missiles, TY-90 AAM. PLAAF deploys at high altitude near LAC.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Z-10_helicopter.jpg/320px-Z-10_helicopter.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Changhe_Z-10",
    },

    # ═══════ PAKISTAN ═══════
    {
        "country": "Pakistan", "system_name": "JF-17 Thunder Block III",
        "system_type": "Fighter_Aircraft", "classification": "Modern",
        "year_inducted": 2022, "tech_generation": 4.5,
        "max_speed_kmph": 1960, "range_km": 1352, "max_altitude_m": 16000,
        "stealth_rating": 3.0, "ew_capability": 7.0, "payload_kg": 3629,
        "reliability": 80, "cost_million_usd": 32, "threat_level": 7.5,
        "operational_status": "Active", "combat_proven": True,
        "description": "Joint Pakistan-China fighter. Block III has AESA radar, HMDS, dual WSO. Carries PL-15 BVR missile. 23 Block IIIs delivered. Claimed to shoot down Indian aircraft in 2019.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/JF-17_Thunder_PAF.jpg/320px-JF-17_Thunder_PAF.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/CAC/PAC_JF-17_Thunder",
    },
    {
        "country": "Pakistan", "system_name": "F-16 Block 52+",
        "system_type": "Fighter_Aircraft", "classification": "Modern",
        "year_inducted": 2010, "tech_generation": 4.5,
        "max_speed_kmph": 2120, "range_km": 1600, "max_altitude_m": 15240,
        "stealth_rating": 2.5, "ew_capability": 7.0, "payload_kg": 7700,
        "reliability": 85, "cost_million_usd": 62, "threat_level": 7.8,
        "operational_status": "Active", "combat_proven": True,
        "description": "Pakistan's most capable fighter. AN/APG-68(V)9 radar, AIM-120C-5 AMRAAM. Used in Balochistan operations. 76 F-16s in PAF. Major US leverage point.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/F-16_June_2008.jpg/320px-F-16_June_2008.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/General_Dynamics_F-16_Fighting_Falcon",
    },
    {
        "country": "Pakistan", "system_name": "Mirage III/V (Pakistan)",
        "system_type": "Fighter_Aircraft", "classification": "Traditional",
        "year_inducted": 1967, "tech_generation": 3.0,
        "max_speed_kmph": 2350, "range_km": 1300, "max_altitude_m": 17000,
        "stealth_rating": 1.5, "ew_capability": 3.5, "payload_kg": 4500,
        "reliability": 55, "cost_million_usd": 12, "threat_level": 4.5,
        "operational_status": "Active", "combat_proven": True,
        "description": "Aging French-origin fighters. Modified with Chinese avionics, Atar 9C engines. Still form significant PAF strength. Used for nuclear strike delivery role.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/Mirage_IIIEA.jpg/320px-Mirage_IIIEA.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Dassault_Mirage_III",
    },
    {
        "country": "Pakistan", "system_name": "Bayraktar TB2 (PAF)",
        "system_type": "UAV_Drone", "classification": "Modern",
        "year_inducted": 2023, "tech_generation": 4.5,
        "max_speed_kmph": 222, "range_km": 300, "max_altitude_m": 8200,
        "stealth_rating": 2.0, "ew_capability": 4.5, "payload_kg": 150,
        "reliability": 78, "cost_million_usd": 5, "threat_level": 6.5,
        "operational_status": "Active", "combat_proven": True,
        "description": "Turkish MALE drone with proven combat record in multiple theatres. Carries MAM-L laser-guided smart micro-munition. Pakistan acquired for ISR and strike roles.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Bayraktar_TB2_Ukraine.jpg/320px-Bayraktar_TB2_Ukraine.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Baykar_Bayraktar_TB2",
    },
    {
        "country": "Pakistan", "system_name": "LY-80 (HQ-16) SAM",
        "system_type": "SAM_System", "classification": "Modern",
        "year_inducted": 2015, "tech_generation": 4.0,
        "max_speed_kmph": 4000, "range_km": 40, "max_altitude_m": 15000,
        "stealth_rating": 0, "ew_capability": 6.0, "payload_kg": 0,
        "reliability": 78, "cost_million_usd": 100, "threat_level": 7.0,
        "operational_status": "Active", "combat_proven": False,
        "description": "Chinese-supplied medium-range SAM system. Active radar homing. Semi-mobile launcher. Complements PAF air defence network. Intended to counter Indian air threats.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/HQ-16_SAM_system.jpg/320px-HQ-16_SAM_system.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/HQ-16",
    },

    # ═══════ NORTH KOREA ═══════
    {
        "country": "North Korea", "system_name": "MiG-29 (KPAAF)",
        "system_type": "Fighter_Aircraft", "classification": "Traditional",
        "year_inducted": 1988, "tech_generation": 3.5,
        "max_speed_kmph": 2445, "range_km": 1430, "max_altitude_m": 18000,
        "stealth_rating": 1.5, "ew_capability": 3.0, "payload_kg": 4500,
        "reliability": 45, "cost_million_usd": 28, "threat_level": 4.5,
        "operational_status": "Active", "combat_proven": False,
        "description": "North Korea's most capable fighter (~35 airframes). Poorly maintained; limited flight hours. Primarily used for symbolic propaganda. Some may lack functional weapons systems.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/da/Mikoyan-Gurevich_MiG-29_%28edit1%29.jpg/320px-Mikoyan-Gurevich_MiG-29_%28edit1%29.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Mikoyan_MiG-29",
    },
    {
        "country": "North Korea", "system_name": "KN-06 SAM System",
        "system_type": "SAM_System", "classification": "Modern",
        "year_inducted": 2017, "tech_generation": 4.0,
        "max_speed_kmph": 10000, "range_km": 150, "max_altitude_m": 25000,
        "stealth_rating": 0, "ew_capability": 5.5, "payload_kg": 0,
        "reliability": 65, "cost_million_usd": 150, "threat_level": 7.0,
        "operational_status": "Active", "combat_proven": False,
        "description": "North Korea's indigenous long-range SAM inspired by S-300/S-400. Claimed 150km range. Limited production. Designed to deter US/South Korean air power.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/KN-06_SAM_North_Korea.jpg/320px-KN-06_SAM_North_Korea.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/KN-06",
    },
    {
        "country": "North Korea", "system_name": "Hwasong-15 ICBM",
        "system_type": "Interceptor_Missile", "classification": "Modern",
        "year_inducted": 2017, "tech_generation": 4.5,
        "max_speed_kmph": 25000, "range_km": 13000, "max_altitude_m": 4475000,
        "stealth_rating": 0, "ew_capability": 3.5, "payload_kg": 1000,
        "reliability": 60, "cost_million_usd": 50, "threat_level": 9.5,
        "operational_status": "Active", "combat_proven": False,
        "description": "North Korea's most powerful ICBM. Can reach all of continental USA. Nuclear-capable. Liquid-fueled (long preparation time). Proliferation risk to India's adversaries.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fb/Hwasong-15.jpg/320px-Hwasong-15.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Hwasong-15",
    },
    {
        "country": "North Korea", "system_name": "Shahed-136 type (NK)",
        "system_type": "UAV_Drone", "classification": "Modern",
        "year_inducted": 2023, "tech_generation": 3.5,
        "max_speed_kmph": 185, "range_km": 2500, "max_altitude_m": 4000,
        "stealth_rating": 3.5, "ew_capability": 2.0, "payload_kg": 50,
        "reliability": 55, "cost_million_usd": 0.02, "threat_level": 6.0,
        "operational_status": "Active", "combat_proven": True,
        "description": "North Korea produces loitering munitions similar to Iranian Shahed-136. Supplied to Russia for Ukraine conflict. Cheap swarm weapons; effective against radar and C2 nodes.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Shahed-136_drone.jpg/320px-Shahed-136_drone.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/HESA_Shahed_136",
    },

    # ═══════ FRANCE ═══════
    {
        "country": "France", "system_name": "Rafale C",
        "system_type": "Fighter_Aircraft", "classification": "Modern",
        "year_inducted": 2006, "tech_generation": 4.5,
        "max_speed_kmph": 1915, "range_km": 3700, "max_altitude_m": 15240,
        "stealth_rating": 4.5, "ew_capability": 9.2, "payload_kg": 9500,
        "reliability": 93, "cost_million_usd": 115, "threat_level": 9.0,
        "operational_status": "Active", "combat_proven": True,
        "description": "France's omnirole 4.5-gen fighter. SPECTRA EW suite (world's best). Meteor BVR, SCALP cruise missile. Combat proven in Libya, Mali, Syria, Iraq.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/Rafale_Farnborough_Airshow_2010.jpg/320px-Rafale_Farnborough_Airshow_2010.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Dassault_Rafale",
    },
    {
        "country": "France", "system_name": "SAMP/T Aster 30",
        "system_type": "SAM_System", "classification": "Modern",
        "year_inducted": 2007, "tech_generation": 5.0,
        "max_speed_kmph": 4500, "range_km": 120, "max_altitude_m": 20000,
        "stealth_rating": 0, "ew_capability": 8.5, "payload_kg": 0,
        "reliability": 90, "cost_million_usd": 380, "threat_level": 8.5,
        "operational_status": "Active", "combat_proven": True,
        "description": "European TLVS air defence. Aster 30 missile (active radar homing), 120km range. Combat proven in Ukraine (supplied to Kyiv). Franco-Italian development.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/SAMP-T_launcher.jpg/320px-SAMP-T_launcher.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Aster_missile_family",
    },
    {
        "country": "France", "system_name": "Tiger HAD Helicopter",
        "system_type": "Helicopter", "classification": "Modern",
        "year_inducted": 2012, "tech_generation": 4.5,
        "max_speed_kmph": 290, "range_km": 800, "max_altitude_m": 4000,
        "stealth_rating": 3.5, "ew_capability": 7.0, "payload_kg": 1300,
        "reliability": 82, "cost_million_usd": 36, "threat_level": 7.8,
        "operational_status": "Active", "combat_proven": True,
        "description": "European attack helicopter. HOT3/Spike missiles, 30mm GIAT cannon. HAD (Support and Destruction) variant highly capable. Used in Mali, Central Africa.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f5/Tiger_HAD_helicopter.jpg/320px-Tiger_HAD_helicopter.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Airbus_Tiger",
    },

    # ═══════ UNITED KINGDOM ═══════
    {
        "country": "United Kingdom", "system_name": "Eurofighter Typhoon FGR4",
        "system_type": "Fighter_Aircraft", "classification": "Modern",
        "year_inducted": 2003, "tech_generation": 4.5,
        "max_speed_kmph": 2495, "range_km": 2900, "max_altitude_m": 19812,
        "stealth_rating": 3.5, "ew_capability": 8.5, "payload_kg": 9000,
        "reliability": 88, "cost_million_usd": 124, "threat_level": 8.8,
        "operational_status": "Active", "combat_proven": True,
        "description": "Multinational 4.5-gen fighter. CAPTOR-E AESA radar, Meteor/AMRAAM. RAF operates 107. Used in Libya, Iraq, Syria. Proposed to India multiple times.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Eurofighter_EF-2000_Typhoon_in_flight_%28cropped%29.jpg/320px-Eurofighter_EF-2000_Typhoon_in_flight_%28cropped%29.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Eurofighter_Typhoon",
    },
    {
        "country": "United Kingdom", "system_name": "F-35B Lightning (UK)",
        "system_type": "Fighter_Aircraft", "classification": "Modern",
        "year_inducted": 2018, "tech_generation": 5.0,
        "max_speed_kmph": 1960, "range_km": 1670, "max_altitude_m": 15240,
        "stealth_rating": 9.0, "ew_capability": 9.8, "payload_kg": 6800,
        "reliability": 85, "cost_million_usd": 135, "threat_level": 9.3,
        "operational_status": "Active", "combat_proven": False,
        "description": "STOVL variant operates from HMS Queen Elizabeth carriers. UK operates 24 F-35B; scaling to 48. Forms centerpiece of UK Carrier Strike Group.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/F-35A_flight_%28cropped%29.jpg/320px-F-35A_flight_%28cropped%29.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Lockheed_Martin_F-35_Lightning_II",
    },
    {
        "country": "United Kingdom", "system_name": "Sky Sabre (CAMM)",
        "system_type": "SAM_System", "classification": "Modern",
        "year_inducted": 2020, "tech_generation": 5.0,
        "max_speed_kmph": 3700, "range_km": 25, "max_altitude_m": 10000,
        "stealth_rating": 0, "ew_capability": 8.0, "payload_kg": 0,
        "reliability": 92, "cost_million_usd": 150, "threat_level": 7.5,
        "operational_status": "Active", "combat_proven": False,
        "description": "Britain's most advanced ground-based air defence. Common Anti-Air Modular Missile. Replaced Rapier. Cold-launch canister means 360° engagement. Supplied to Poland.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Sky_Sabre_CAMM_launcher.jpg/320px-Sky_Sabre_CAMM_launcher.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/CAMM_(missile)",
    },

    # ═══════ ISRAEL ═══════
    {
        "country": "Israel", "system_name": "F-35I Adir",
        "system_type": "Fighter_Aircraft", "classification": "Modern",
        "year_inducted": 2017, "tech_generation": 5.0,
        "max_speed_kmph": 1960, "range_km": 2220, "max_altitude_m": 15240,
        "stealth_rating": 9.5, "ew_capability": 10.0, "payload_kg": 8160,
        "reliability": 92, "cost_million_usd": 110, "threat_level": 9.8,
        "operational_status": "Active", "combat_proven": True,
        "description": "World's most combat-experienced F-35. Israel modified AN/APG-81 with indigenous Israeli systems. First combat use globally. Used for strikes on Syria, Iraq, Iran-backed targets.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/F-35A_flight_%28cropped%29.jpg/320px-F-35A_flight_%28cropped%29.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Lockheed_Martin_F-35_Lightning_II",
    },
    {
        "country": "Israel", "system_name": "Iron Dome",
        "system_type": "SAM_System", "classification": "Modern",
        "year_inducted": 2011, "tech_generation": 4.5,
        "max_speed_kmph": 2800, "range_km": 70, "max_altitude_m": 10000,
        "stealth_rating": 0, "ew_capability": 8.0, "payload_kg": 0,
        "reliability": 90, "cost_million_usd": 100, "threat_level": 8.0,
        "operational_status": "Active", "combat_proven": True,
        "description": "World-famous short-range air defence. 90%+ intercept rate. Has intercepted 3000+ rockets. Tamir interceptors ($50k each). US co-funded. Battle-tested in Gaza conflicts.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Iron_Dome_arrayed_2012.jpg/320px-Iron_Dome_arrayed_2012.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Iron_Dome",
    },
    {
        "country": "Israel", "system_name": "Arrow-3 (Chetz-3)",
        "system_type": "SAM_System", "classification": "Modern",
        "year_inducted": 2017, "tech_generation": 5.5,
        "max_speed_kmph": 14400, "range_km": 2400, "max_altitude_m": 100000,
        "stealth_rating": 0, "ew_capability": 9.0, "payload_kg": 0,
        "reliability": 95, "cost_million_usd": 3000, "threat_level": 9.5,
        "operational_status": "Active", "combat_proven": True,
        "description": "Exo-atmospheric interceptor for ballistic missiles. Hit-to-kill technology outside atmosphere. Intercepted Iranian missiles April 2024. Being offered to India.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e2/Arrow_3_missile_system.jpg/320px-Arrow_3_missile_system.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Arrow_3",
    },
    {
        "country": "Israel", "system_name": "Heron TP (IAI)",
        "system_type": "UAV_Drone", "classification": "Modern",
        "year_inducted": 2010, "tech_generation": 4.5,
        "max_speed_kmph": 220, "range_km": 1500, "max_altitude_m": 14000,
        "stealth_rating": 2.5, "ew_capability": 6.5, "payload_kg": 1000,
        "reliability": 92, "cost_million_usd": 35, "threat_level": 7.0,
        "operational_status": "Active", "combat_proven": True,
        "description": "Israel's strategic MALE UCAV. 52-hour endurance, 1000kg payload. SAR, EO/IR sensors. India leases and operates Heron TP for LAC surveillance and strike.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Heron_TP_UAV.jpg/320px-Heron_TP_UAV.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/IAI_Heron_TP",
    },
    {
        "country": "Israel", "system_name": "David's Sling",
        "system_type": "SAM_System", "classification": "Modern",
        "year_inducted": 2017, "tech_generation": 5.0,
        "max_speed_kmph": 7560, "range_km": 300, "max_altitude_m": 15000,
        "stealth_rating": 0, "ew_capability": 9.5, "payload_kg": 0,
        "reliability": 93, "cost_million_usd": 1000, "threat_level": 9.2,
        "operational_status": "Active", "combat_proven": True,
        "description": "Fills the gap between Iron Dome and Arrow. Stunner interceptor (hit-to-kill + fragmentation). Range 40-300km. Battle-tested against Iranian ballistic missile salvo in April 2024.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/David%27s_Sling_system.jpg/320px-David%27s_Sling_system.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/David%27s_Sling",
    },

    # ═══════ JAPAN ═══════
    {
        "country": "Japan", "system_name": "F-35A (JASDF)",
        "system_type": "Fighter_Aircraft", "classification": "Modern",
        "year_inducted": 2019, "tech_generation": 5.0,
        "max_speed_kmph": 1960, "range_km": 2220, "max_altitude_m": 15240,
        "stealth_rating": 9.0, "ew_capability": 9.8, "payload_kg": 8160,
        "reliability": 88, "cost_million_usd": 135, "threat_level": 9.5,
        "operational_status": "Active", "combat_proven": False,
        "description": "JASDF acquiring 147 F-35s (105 F-35A + 42 F-35B). Replaces aging F-15J fleet. Largest non-US F-35 customer. Primary deterrent against PLAAF and KPAAF.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/F-35A_flight_%28cropped%29.jpg/320px-F-35A_flight_%28cropped%29.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Lockheed_Martin_F-35_Lightning_II",
    },
    {
        "country": "Japan", "system_name": "F-15J Kai (Modernized)",
        "system_type": "Fighter_Aircraft", "classification": "Traditional",
        "year_inducted": 1981, "tech_generation": 4.0,
        "max_speed_kmph": 3017, "range_km": 4630, "max_altitude_m": 20000,
        "stealth_rating": 2.5, "ew_capability": 6.5, "payload_kg": 10659,
        "reliability": 80, "cost_million_usd": 55, "threat_level": 7.5,
        "operational_status": "Active", "combat_proven": False,
        "description": "Japan-specific F-15. Upgraded with IRST, J/ALQ-2 EW. Being further upgraded to 'F-15JSI' standard with AESA radar. 155 in service. Main interceptor role.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/eb/F-15EX_first_flight.jpg/320px-F-15EX_first_flight.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/McDonnell_Douglas_F-15_Eagle",
    },
    {
        "country": "Japan", "system_name": "PAC-3 MSE (JASDF)",
        "system_type": "SAM_System", "classification": "Modern",
        "year_inducted": 2019, "tech_generation": 5.0,
        "max_speed_kmph": 8100, "range_km": 35, "max_altitude_m": 24000,
        "stealth_rating": 0, "ew_capability": 8.5, "payload_kg": 0,
        "reliability": 95, "cost_million_usd": 4, "threat_level": 8.8,
        "operational_status": "Active", "combat_proven": False,
        "description": "Japan's primary TBM defense system. Deployed at 6 JASDF bases. Upgraded to MSE (Missile Segment Enhancement) for extended range. Complemented by Aegis BMD ships.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/Patriot_missile_launch_b.jpg/320px-Patriot_missile_launch_b.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Patriot_missile_system",
    },

    # ═══════ AUSTRALIA ═══════
    {
        "country": "Australia", "system_name": "F-35A (RAAF)",
        "system_type": "Fighter_Aircraft", "classification": "Modern",
        "year_inducted": 2021, "tech_generation": 5.0,
        "max_speed_kmph": 1960, "range_km": 2220, "max_altitude_m": 15240,
        "stealth_rating": 9.0, "ew_capability": 9.8, "payload_kg": 8160,
        "reliability": 90, "cost_million_usd": 135, "threat_level": 9.5,
        "operational_status": "Active", "combat_proven": False,
        "description": "Australia's next-gen fighter. 72 ordered. Replaces legacy Hornet. RAAF pilots training alongside USAF. Integrated into US-AU network-centric warfare framework.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/F-35A_flight_%28cropped%29.jpg/320px-F-35A_flight_%28cropped%29.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Lockheed_Martin_F-35_Lightning_II",
    },
    {
        "country": "Australia", "system_name": "F/A-18F Super Hornet",
        "system_type": "Fighter_Aircraft", "classification": "Modern",
        "year_inducted": 2010, "tech_generation": 4.5,
        "max_speed_kmph": 1900, "range_km": 1275, "max_altitude_m": 15240,
        "stealth_rating": 3.0, "ew_capability": 8.0, "payload_kg": 8050,
        "reliability": 88, "cost_million_usd": 67, "threat_level": 8.2,
        "operational_status": "Active", "combat_proven": True,
        "description": "Australia operates 24 F/A-18F and 12 EA-18G Growler EW aircraft. Used in Middle East operations. Being replaced by F-35 but retained for Growler-unique EW role.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/FA-18F_Super_Hornet_VFA-11_2012.jpg/320px-FA-18F_Super_Hornet_VFA-11_2012.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Boeing_F/A-18E/F_Super_Hornet",
    },

    # ═══════ IRAN ═══════
    {
        "country": "Iran", "system_name": "F-14A Tomcat (IRIAF)",
        "system_type": "Fighter_Aircraft", "classification": "Traditional",
        "year_inducted": 1976, "tech_generation": 3.5,
        "max_speed_kmph": 2485, "range_km": 3200, "max_altitude_m": 16150,
        "stealth_rating": 1.5, "ew_capability": 3.0, "payload_kg": 6577,
        "reliability": 38, "cost_million_usd": 18, "threat_level": 4.5,
        "operational_status": "Active", "combat_proven": True,
        "description": "Iconic swing-wing fighter, US-supplied in 1970s. Only ~10-20 flyable. Spare parts crisis. AWG-9 radar still impressive; AIM-54 Phoenix missiles. Symbol of Iranian air power.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/US_Navy_040811-N-7559H-007_F-14_Tomcat.jpg/320px-US_Navy_040811-N-7559H-007_F-14_Tomcat.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Grumman_F-14_Tomcat",
    },
    {
        "country": "Iran", "system_name": "Bavar-373 SAM",
        "system_type": "SAM_System", "classification": "Modern",
        "year_inducted": 2019, "tech_generation": 4.5,
        "max_speed_kmph": 14400, "range_km": 300, "max_altitude_m": 27000,
        "stealth_rating": 0, "ew_capability": 7.0, "payload_kg": 0,
        "reliability": 72, "cost_million_usd": 200, "threat_level": 7.5,
        "operational_status": "Active", "combat_proven": False,
        "description": "Iran's domestic answer to S-300. Phased-array AESA radar, 300km range claimed. Developed after Russia delayed S-300 delivery. Actual performance unverified.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Bavar-373_SAM_Iran.jpg/320px-Bavar-373_SAM_Iran.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Bavar-373",
    },
    {
        "country": "Iran", "system_name": "Shahed-136 (Loitering Munition)",
        "system_type": "UAV_Drone", "classification": "Modern",
        "year_inducted": 2021, "tech_generation": 3.5,
        "max_speed_kmph": 185, "range_km": 2500, "max_altitude_m": 4000,
        "stealth_rating": 3.5, "ew_capability": 2.0, "payload_kg": 50,
        "reliability": 60, "cost_million_usd": 0.02, "threat_level": 6.0,
        "operational_status": "Active", "combat_proven": True,
        "description": "Low-cost kamikaze drone (Geran-2 in Russian use). Delta-wing, turbojet. Supplied 2000+ to Russia for Ukraine. India monitoring proliferation to Pakistan. GPS+inertial.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Shahed-136_drone.jpg/320px-Shahed-136_drone.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/HESA_Shahed_136",
    },

    # ═══════ TURKEY ═══════
    {
        "country": "Turkey", "system_name": "KAAN (TF-X)",
        "system_type": "Fighter_Aircraft", "classification": "Modern",
        "year_inducted": 2025, "tech_generation": 5.0,
        "max_speed_kmph": 2200, "range_km": 2800, "max_altitude_m": 18000,
        "stealth_rating": 6.5, "ew_capability": 7.5, "payload_kg": 6000,
        "reliability": 65, "cost_million_usd": 100, "threat_level": 8.0,
        "operational_status": "Active", "combat_proven": False,
        "description": "Turkey's indigenously developed 5th-gen fighter. First flew 2023. Twin-engine, internal weapons bay. F110-GE engines (US). Future domestic engine under development.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/KAAN_fighter_jet_Turkey.jpg/320px-KAAN_fighter_jet_Turkey.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/TAI_KAAN",
    },
    {
        "country": "Turkey", "system_name": "Bayraktar AKINCI",
        "system_type": "UAV_Drone", "classification": "Modern",
        "year_inducted": 2021, "tech_generation": 5.0,
        "max_speed_kmph": 361, "range_km": 1500, "max_altitude_m": 12000,
        "stealth_rating": 3.5, "ew_capability": 6.5, "payload_kg": 1350,
        "reliability": 82, "cost_million_usd": 75, "threat_level": 8.5,
        "operational_status": "Active", "combat_proven": True,
        "description": "Turkey's HALE combat drone. Twin-turboprop, air-to-air missiles (Göktuğ), SOM cruise missile. Phased array radar. Game-changer in drone warfare. Exported to multiple countries.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Bayraktar_Akinci_UCAV.jpg/320px-Bayraktar_Akinci_UCAV.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Baykar_Akıncı",
    },
    {
        "country": "Turkey", "system_name": "F-16 Block 50+ (TurAF)",
        "system_type": "Fighter_Aircraft", "classification": "Modern",
        "year_inducted": 2006, "tech_generation": 4.5,
        "max_speed_kmph": 2120, "range_km": 1600, "max_altitude_m": 15240,
        "stealth_rating": 2.5, "ew_capability": 7.5, "payload_kg": 7700,
        "reliability": 87, "cost_million_usd": 62, "threat_level": 7.8,
        "operational_status": "Active", "combat_proven": True,
        "description": "Turkey's primary combat aircraft (245 F-16s). Block 50+ with APG-68 radar. Extensively used over Syria. Upgraded to Block 70 standard. Excluded from F-35 program over S-400 purchase.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/F-16_June_2008.jpg/320px-F-16_June_2008.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/General_Dynamics_F-16_Fighting_Falcon",
    },

    # ═══════ SAUDI ARABIA ═══════
    {
        "country": "Saudi Arabia", "system_name": "F-15SA Eagle",
        "system_type": "Fighter_Aircraft", "classification": "Modern",
        "year_inducted": 2016, "tech_generation": 4.5,
        "max_speed_kmph": 3017, "range_km": 3900, "max_altitude_m": 20000,
        "stealth_rating": 3.0, "ew_capability": 8.5, "payload_kg": 13381,
        "reliability": 88, "cost_million_usd": 87, "threat_level": 8.5,
        "operational_status": "Active", "combat_proven": True,
        "description": "Most advanced F-15 variant when delivered. AN/APG-63(V)3 AESA, DEWS EW suite, dual-seat. 84 on order. Used in Yemen strikes. Technically superior to most regional opponents.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/eb/F-15EX_first_flight.jpg/320px-F-15EX_first_flight.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/McDonnell_Douglas_F-15_Eagle",
    },
    {
        "country": "Saudi Arabia", "system_name": "Eurofighter Typhoon (RSAF)",
        "system_type": "Fighter_Aircraft", "classification": "Modern",
        "year_inducted": 2009, "tech_generation": 4.5,
        "max_speed_kmph": 2495, "range_km": 2900, "max_altitude_m": 19812,
        "stealth_rating": 3.5, "ew_capability": 8.0, "payload_kg": 9000,
        "reliability": 82, "cost_million_usd": 124, "threat_level": 8.5,
        "operational_status": "Active", "combat_proven": True,
        "description": "Saudi Arabia's 72 Typhoons (Project Salam). Used in Yemen air campaign. CAPTOR radar, Storm Shadow cruise missiles, Brimstone anti-tank missiles.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Eurofighter_EF-2000_Typhoon_in_flight_%28cropped%29.jpg/320px-Eurofighter_EF-2000_Typhoon_in_flight_%28cropped%29.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Eurofighter_Typhoon",
    },
    {
        "country": "Saudi Arabia", "system_name": "Patriot PAC-3 (RSAF)",
        "system_type": "SAM_System", "classification": "Modern",
        "year_inducted": 2015, "tech_generation": 5.0,
        "max_speed_kmph": 8100, "range_km": 35, "max_altitude_m": 24000,
        "stealth_rating": 0, "ew_capability": 8.5, "payload_kg": 0,
        "reliability": 82, "cost_million_usd": 4, "threat_level": 8.0,
        "operational_status": "Active", "combat_proven": True,
        "description": "Saudi Arabia operates 15+ Patriot batteries. Tested against Iranian/Houthi ballistic and cruise missiles with mixed results. Major US arms sale controversy.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/Patriot_missile_launch_b.jpg/320px-Patriot_missile_launch_b.jpg",
        "wikipedia_url": "https://en.wikipedia.org/wiki/Patriot_missile_system",
    },
]


# ─────────────────────────────────────────────
# 3.  CONFLICT SCENARIOS  (for War Prediction ML)
#     Synthetic data representing historical
#     and hypothetical air combat outcomes
# ─────────────────────────────────────────────
def compute_force_score(country_name, systems_df):
    c = systems_df[systems_df["country"] == country_name]
    if len(c) == 0:
        return {}

    return {
        "total_systems": len(c),
        "modern_pct": round((c["classification"] == "Modern").mean() * 100, 1),
        "avg_threat_level": round(c["threat_level"].mean(), 2),
        "avg_stealth": round(c["stealth_rating"].mean(), 2),
        "avg_ew": round(c["ew_capability"].mean(), 2),
        "avg_reliability": round(c["reliability"].mean(), 1),
        "avg_tech_gen": round(c["tech_generation"].mean(), 2),
        "fighter_count": len(c[c["system_type"] == "Fighter_Aircraft"]),
        "sam_count": len(c[c["system_type"] == "SAM_System"]),
        "uav_count": len(c[c["system_type"] == "UAV_Drone"]),
    }


def generate_conflict_scenarios(systems_df, countries_df, n_scenarios=700):
    rng = np.random.default_rng(42)
    country_list = countries_df["country"].tolist()
    scenarios = []

    force_cache = {c: compute_force_score(c, systems_df) for c in country_list}
    zone_map = {"Red": 3, "Yellow": 2, "Green": 1}

    for i in range(n_scenarios):
        att, dfn = rng.choice(country_list, size=2, replace=False)

        att_row = countries_df[countries_df["country"] == att].iloc[0]
        dfn_row = countries_df[countries_df["country"] == dfn].iloc[0]

        att_fs = force_cache[att]
        dfn_fs = force_cache[dfn]

        if not att_fs or not dfn_fs:
            continue

        threat_ratio = att_fs["avg_threat_level"] / max(dfn_fs["avg_threat_level"], 0.1)
        tech_ratio = att_fs["avg_tech_gen"] / max(dfn_fs["avg_tech_gen"], 0.1)
        number_ratio = att_row["combat_aircraft_count"] / max(dfn_row["combat_aircraft_count"], 1)
        budget_ratio = att_row["military_budget_billion_usd"] / max(dfn_row["military_budget_billion_usd"], 0.1)

        advantage = float(np.clip(50 + (threat_ratio + tech_ratio + number_ratio + budget_ratio) * 2, 5, 95))
        noisy_adv = float(np.clip(advantage + rng.normal(0, 8), 5, 95))

        if noisy_adv > 62:
            outcome = "Attacker_Wins"
            win_prob = round(float(np.clip(noisy_adv / 100, 0.55, 0.95)), 2)
        elif noisy_adv < 38:
            outcome = "Defender_Wins"
            win_prob = round(float(np.clip(noisy_adv / 100, 0.05, 0.45)), 2)
        else:
            outcome = "Stalemate"
            win_prob = round(float(np.clip(noisy_adv / 100, 0.40, 0.60)), 2)

        scenarios.append({
            "scenario_id": f"SCN_{i+1:04d}",
            "attacker": att,
            "defender": dfn,
            "att_avg_threat": att_fs["avg_threat_level"],
            "att_avg_tech_gen": att_fs["avg_tech_gen"],
            "att_modern_pct": att_fs["modern_pct"],
            "att_avg_stealth": att_fs["avg_stealth"],
            "att_avg_ew": att_fs["avg_ew"],
            "att_fighter_count": att_fs["fighter_count"],
            "att_sam_count": att_fs["sam_count"],
            "att_uav_count": att_fs["uav_count"],
            "att_military_budget": att_row["military_budget_billion_usd"],
            "att_aircraft_count": att_row["combat_aircraft_count"],
            "att_zone": zone_map[att_row["risk_zone"]],
            "dfn_avg_threat": dfn_fs["avg_threat_level"],
            "dfn_avg_tech_gen": dfn_fs["avg_tech_gen"],
            "dfn_modern_pct": dfn_fs["modern_pct"],
            "dfn_avg_stealth": dfn_fs["avg_stealth"],
            "dfn_avg_ew": dfn_fs["avg_ew"],
            "dfn_fighter_count": dfn_fs["fighter_count"],
            "dfn_sam_count": dfn_fs["sam_count"],
            "dfn_uav_count": dfn_fs["uav_count"],
            "dfn_military_budget": dfn_row["military_budget_billion_usd"],
            "dfn_aircraft_count": dfn_row["combat_aircraft_count"],
            "dfn_zone": zone_map[dfn_row["risk_zone"]],
            "threat_ratio": round(threat_ratio, 3),
            "tech_ratio": round(tech_ratio, 3),
            "number_ratio": round(number_ratio, 3),
            "budget_ratio": round(min(budget_ratio, 100), 3),
            "outcome": outcome,
            "attacker_win_probability": win_prob
        })

    return pd.DataFrame(scenarios)


# ─────────────────────────────────────────────
# 4.  MAIN — BUILD & SAVE EVERYTHING
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import os

    # Get project root (one level above src/)
    

    # ── Build DataFrames ──────────────────────
    countries_df = pd.DataFrame(COUNTRIES)

    # ── Add legacy/traditional systems per country ──
    # (ensures ML classifier has enough traditional examples)
    LEGACY_SYSTEMS = [
        # India legacy
        {"country":"India","system_name":"MiG-21 Bison","system_type":"Fighter_Aircraft","classification":"Traditional","year_inducted":1963,"tech_generation":2.5,"max_speed_kmph":2175,"range_km":1210,"max_altitude_m":19000,"stealth_rating":1.0,"ew_capability":3.5,"payload_kg":2000,"reliability":52,"cost_million_usd":12,"threat_level":3.5,"operational_status":"Active","combat_proven":True,"description":"India's backbone fighter for decades. Nicknamed 'Flying Coffin' due to high accident rate. 200+ lost in crashes. Being retired.","image_url":"https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Mig21_2.jpg/320px-Mig21_2.jpg","wikipedia_url":"https://en.wikipedia.org/wiki/Mikoyan-Gurevich_MiG-21"},
        {"country":"India","system_name":"SEPECAT Jaguar IS","system_type":"Fighter_Aircraft","classification":"Traditional","year_inducted":1979,"tech_generation":3.0,"max_speed_kmph":1700,"range_km":1408,"max_altitude_m":14000,"stealth_rating":1.5,"ew_capability":3.0,"payload_kg":4763,"reliability":60,"cost_million_usd":18,"threat_level":4.0,"operational_status":"Active","combat_proven":True,"description":"Anglo-French ground attack aircraft. Used in Kargil. Being replaced. IAF upgrades include DARIN-III avionics, Israeli EW.","image_url":"https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/SEPECAT_Jaguar.jpg/320px-SEPECAT_Jaguar.jpg","wikipedia_url":"https://en.wikipedia.org/wiki/SEPECAT_Jaguar"},
        # USA legacy
        {"country":"USA","system_name":"A-10 Thunderbolt II","system_type":"Fighter_Aircraft","classification":"Traditional","year_inducted":1977,"tech_generation":3.0,"max_speed_kmph":706,"range_km":1200,"max_altitude_m":13700,"stealth_rating":0.5,"ew_capability":4.0,"payload_kg":7260,"reliability":78,"cost_million_usd":19,"threat_level":5.5,"operational_status":"Active","combat_proven":True,"description":"Iconic tank-buster. GAU-8 30mm cannon. 281 in USAF. Survived multiple retirement attempts. Proven in Gulf War, Bosnia, Iraq, Afghanistan.","image_url":"https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/A-10_Thunderbolt_II_In-flight-2.jpg/320px-A-10_Thunderbolt_II_In-flight-2.jpg","wikipedia_url":"https://en.wikipedia.org/wiki/Fairchild_Republic_A-10_Thunderbolt_II"},
        {"country":"USA","system_name":"MIM-23 HAWK SAM","system_type":"SAM_System","classification":"Traditional","year_inducted":1960,"tech_generation":2.0,"max_speed_kmph":3200,"range_km":45,"max_altitude_m":11000,"stealth_rating":0,"ew_capability":4.0,"payload_kg":0,"reliability":65,"cost_million_usd":8,"threat_level":5.0,"operational_status":"Reserve","combat_proven":True,"description":"Cold War era SAM still in service with many allies. Replaced by Patriot in US service. Used by Israel, Saudi Arabia, Taiwan, many NATO members.","image_url":"https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/HAWK_missile_system.jpg/320px-HAWK_missile_system.jpg","wikipedia_url":"https://en.wikipedia.org/wiki/MIM-23_Hawk"},
        # Russia legacy
        {"country":"Russia","system_name":"MiG-31BM Foxhound","system_type":"Fighter_Aircraft","classification":"Traditional","year_inducted":1981,"tech_generation":3.5,"max_speed_kmph":3000,"range_km":3000,"max_altitude_m":20600,"stealth_rating":1.5,"ew_capability":5.0,"payload_kg":9000,"reliability":72,"cost_million_usd":45,"threat_level":6.0,"operational_status":"Active","combat_proven":False,"description":"World's fastest interceptor. Zaslon phased-array radar. Carries Kh-47M2 Kinzhal hypersonic missile. 105 in VKS. Supersonic at low altitude.","image_url":"https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/MiG-31.jpg/320px-MiG-31.jpg","wikipedia_url":"https://en.wikipedia.org/wiki/Mikoyan_MiG-31"},
        {"country":"Russia","system_name":"S-300V4 (SA-23)","system_type":"SAM_System","classification":"Traditional","year_inducted":1983,"tech_generation":3.5,"max_speed_kmph":14400,"range_km":400,"max_altitude_m":27000,"stealth_rating":0,"ew_capability":7.0,"payload_kg":0,"reliability":82,"cost_million_usd":300,"threat_level":8.0,"operational_status":"Active","combat_proven":True,"description":"Latest S-300V variant for ballistic and cruise missile defence. 400km range. Used to defend critical Russian assets. Export variant sold to India as S-300PMU-2.","image_url":"https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/S-300_system.jpg/320px-S-300_system.jpg","wikipedia_url":"https://en.wikipedia.org/wiki/S-300_missile_system"},
        # China legacy
        {"country":"China","system_name":"J-7G (MiG-21 derivative)","system_type":"Fighter_Aircraft","classification":"Traditional","year_inducted":1966,"tech_generation":2.5,"max_speed_kmph":2175,"range_km":1350,"max_altitude_m":18500,"stealth_rating":0.5,"ew_capability":2.5,"payload_kg":1000,"reliability":55,"cost_million_usd":6,"threat_level":3.0,"operational_status":"Reserve","combat_proven":True,"description":"China's long-produced MiG-21 clone. Final J-7G variant had pulse-Doppler radar. Widely exported (Pakistan's F-7PG). 150+ still in reserve PLAAF.","image_url":"https://upload.wikimedia.org/wikipedia/commons/thumb/f/f2/Chengdu_J-7.jpg/320px-Chengdu_J-7.jpg","wikipedia_url":"https://en.wikipedia.org/wiki/Chengdu_J-7"},
        {"country":"China","system_name":"HQ-2 SAM (SA-2 variant)","system_type":"SAM_System","classification":"Traditional","year_inducted":1967,"tech_generation":2.0,"max_speed_kmph":4300,"range_km":35,"max_altitude_m":27000,"stealth_rating":0,"ew_capability":2.0,"payload_kg":0,"reliability":50,"cost_million_usd":5,"threat_level":4.0,"operational_status":"Reserve","combat_proven":True,"description":"China's indigenous SA-2 copy. Improved guidance over Soviet original. Shot down U-2 spy planes. Still held in reserve. Widely exported to North Korea and other states.","image_url":"https://upload.wikimedia.org/wikipedia/commons/thumb/7/77/HQ-2_SAM_system.jpg/320px-HQ-2_SAM_system.jpg","wikipedia_url":"https://en.wikipedia.org/wiki/HQ-2"},
        # Pakistan legacy
        {"country":"Pakistan","system_name":"F-7PG (J-7 export)","system_type":"Fighter_Aircraft","classification":"Traditional","year_inducted":2002,"tech_generation":3.0,"max_speed_kmph":2175,"range_km":1350,"max_altitude_m":18000,"stealth_rating":0.5,"ew_capability":3.0,"payload_kg":1000,"reliability":58,"cost_million_usd":8,"threat_level":3.5,"operational_status":"Active","combat_proven":False,"description":"Chinese-supplied MiG-21 derivative. Grifo-7 radar, HOTAS cockpit. PAF operates 50+. Primarily for air defence. Being replaced by JF-17.","image_url":"https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/JF-17_Thunder_PAF.jpg/320px-JF-17_Thunder_PAF.jpg","wikipedia_url":"https://en.wikipedia.org/wiki/Chengdu_J-7"},
        # North Korea legacy
        {"country":"North Korea","system_name":"MiG-21 Fishbed (KPAAF)","system_type":"Fighter_Aircraft","classification":"Traditional","year_inducted":1965,"tech_generation":2.5,"max_speed_kmph":2175,"range_km":1210,"max_altitude_m":19000,"stealth_rating":0.5,"ew_capability":1.5,"payload_kg":1000,"reliability":35,"cost_million_usd":5,"threat_level":2.5,"operational_status":"Active","combat_proven":False,"description":"KPAAF operates 120+ MiG-21s of various sub-types. Extreme maintenance challenges, likely only fraction are flyable. Pilots severely limited in flight hours due to fuel shortages.","image_url":"https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Mig21_2.jpg/320px-Mig21_2.jpg","wikipedia_url":"https://en.wikipedia.org/wiki/Mikoyan-Gurevich_MiG-21"},
        # France legacy
        {"country":"France","system_name":"Mirage 2000-C","system_type":"Fighter_Aircraft","classification":"Traditional","year_inducted":1984,"tech_generation":3.5,"max_speed_kmph":2340,"range_km":1850,"max_altitude_m":17060,"stealth_rating":2.0,"ew_capability":4.5,"payload_kg":6300,"reliability":72,"cost_million_usd":42,"threat_level":5.5,"operational_status":"Reserve","combat_proven":True,"description":"Air defence variant of Mirage 2000 family. Replaced by Rafale in French service. 124 built for France. Still in service with India, Egypt, UAE, Greece.","image_url":"https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/Mirage_2000_img_3924.jpg/320px-Mirage_2000_img_3924.jpg","wikipedia_url":"https://en.wikipedia.org/wiki/Dassault_Mirage_2000"},
        # UK legacy
        {"country":"United Kingdom","system_name":"Panavia Tornado GR4","system_type":"Fighter_Aircraft","classification":"Traditional","year_inducted":1974,"tech_generation":3.0,"max_speed_kmph":2337,"range_km":1390,"max_altitude_m":15240,"stealth_rating":1.5,"ew_capability":4.5,"payload_kg":9000,"reliability":65,"cost_million_usd":28,"threat_level":5.0,"operational_status":"Decommissioned","combat_proven":True,"description":"Retired in 2019. Variable-sweep wing strike aircraft. Used in Gulf War, Bosnia, Kosovo, Afghanistan, Libya, Iraq. Royal Saudi Air Force still operates Tornado.","image_url":"https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/Tornado_GR4_RAF.jpg/320px-Tornado_GR4_RAF.jpg","wikipedia_url":"https://en.wikipedia.org/wiki/Panavia_Tornado"},
        # Israel legacy
        {"country":"Israel","system_name":"F-16I Sufa","system_type":"Fighter_Aircraft","classification":"Traditional","year_inducted":2003,"tech_generation":4.0,"max_speed_kmph":2120,"range_km":1600,"max_altitude_m":15240,"stealth_rating":2.0,"ew_capability":6.5,"payload_kg":7700,"reliability":88,"cost_million_usd":45,"threat_level":7.0,"operational_status":"Active","combat_proven":True,"description":"Israel-specific F-16D with conformal fuel tanks and Israeli systems. 102 in IAF service. Used extensively for long-range strike missions. Carries Python-5, Derby missiles.","image_url":"https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/F-16_June_2008.jpg/320px-F-16_June_2008.jpg","wikipedia_url":"https://en.wikipedia.org/wiki/General_Dynamics_F-16_Fighting_Falcon"},
        # Japan legacy
        {"country":"Japan","system_name":"Mitsubishi F-2","system_type":"Fighter_Aircraft","classification":"Traditional","year_inducted":2000,"tech_generation":4.0,"max_speed_kmph":2124,"range_km":2900,"max_altitude_m":18000,"stealth_rating":2.5,"ew_capability":6.0,"payload_kg":8085,"reliability":82,"cost_million_usd":105,"threat_level":6.5,"operational_status":"Active","combat_proven":False,"description":"Japanese F-16 derivative with AESA radar and Japanese systems. 98 in service. Primary anti-ship role with ASM-2. Being supplemented by F-35.","image_url":"https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Mitsubishi_F-2.jpg/320px-Mitsubishi_F-2.jpg","wikipedia_url":"https://en.wikipedia.org/wiki/Mitsubishi_F-2"},
        # Iran legacy
        {"country":"Iran","system_name":"MiG-29 (IRIAF)","system_type":"Fighter_Aircraft","classification":"Traditional","year_inducted":1990,"tech_generation":3.5,"max_speed_kmph":2445,"range_km":1430,"max_altitude_m":18000,"stealth_rating":1.5,"ew_capability":3.5,"payload_kg":4500,"reliability":42,"cost_million_usd":22,"threat_level":4.5,"operational_status":"Active","combat_proven":False,"description":"Acquired after Gulf War. Iran operates ~35 MiG-29s. Spare parts crisis limits readiness. Radar upgrade attempted but limited by sanctions. R-73 short-range missiles.","image_url":"https://upload.wikimedia.org/wikipedia/commons/thumb/d/da/Mikoyan-Gurevich_MiG-29_%28edit1%29.jpg/320px-Mikoyan-Gurevich_MiG-29_%28edit1%29.jpg","wikipedia_url":"https://en.wikipedia.org/wiki/Mikoyan_MiG-29"},
        # Turkey legacy
        {"country":"Turkey","system_name":"F-4E Phantom (TurAF)","system_type":"Fighter_Aircraft","classification":"Traditional","year_inducted":1974,"tech_generation":2.5,"max_speed_kmph":2370,"range_km":1145,"max_altitude_m":18300,"stealth_rating":0.5,"ew_capability":3.0,"payload_kg":8480,"reliability":50,"cost_million_usd":14,"threat_level":3.5,"operational_status":"Decommissioned","combat_proven":True,"description":"Turkish Phantoms retired in 2018. Underwent Terminator 2020 upgrade with Israeli Elta EL/M-2032 radar. Used for decades in NATO and regional missions.","image_url":"https://upload.wikimedia.org/wikipedia/commons/thumb/6/64/F-4_Phantom_II_%28cropped%29.jpg/320px-F-4_Phantom_II_%28cropped%29.jpg","wikipedia_url":"https://en.wikipedia.org/wiki/McDonnell_Douglas_F-4_Phantom_II"},
        # Saudi legacy
        {"country":"Saudi Arabia","system_name":"Tornado IDS (RSAF)","system_type":"Fighter_Aircraft","classification":"Traditional","year_inducted":1986,"tech_generation":3.0,"max_speed_kmph":2337,"range_km":1390,"max_altitude_m":15240,"stealth_rating":1.0,"ew_capability":4.0,"payload_kg":9000,"reliability":60,"cost_million_usd":28,"threat_level":5.0,"operational_status":"Active","combat_proven":True,"description":"RSAF operates 80+ Tornados. Extensively used in Yemen air campaign. Al Yamamah deal controversy. Being phased out in favour of Typhoon/F-15SA.","image_url":"https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/Tornado_GR4_RAF.jpg/320px-Tornado_GR4_RAF.jpg","wikipedia_url":"https://en.wikipedia.org/wiki/Panavia_Tornado"},
        # Australia legacy
        {"country":"Australia","system_name":"F/A-18A Classic Hornet","system_type":"Fighter_Aircraft","classification":"Traditional","year_inducted":1985,"tech_generation":3.5,"max_speed_kmph":1915,"range_km":1065,"max_altitude_m":15240,"stealth_rating":1.5,"ew_capability":5.0,"payload_kg":7031,"reliability":75,"cost_million_usd":30,"threat_level":5.5,"operational_status":"Decommissioned","combat_proven":True,"description":"RAAF's primary fighter for 30+ years. Retired in 2021 when F-35A entered service. Used in Gulf War, East Timor operations. Highly regarded by pilots.","image_url":"https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/F-18_Hornet_DoD.jpg/320px-F-18_Hornet_DoD.jpg","wikipedia_url":"https://en.wikipedia.org/wiki/McDonnell_Douglas_F/A-18_Hornet"},
    ]

    systems_df = pd.DataFrame(SYSTEMS + LEGACY_SYSTEMS)

    # Add derived columns to systems
    systems_df["fuel_efficiency"] = systems_df.apply(
        lambda r: round(np.random.uniform(0.5, 2.0) if r["classification"] == "Modern"
                        else np.random.uniform(0.2, 0.8), 3), axis=1)
    systems_df["export_available"] = systems_df.apply(
        lambda r: np.random.choice(["Yes", "No"], p=[0.55, 0.45]), axis=1)
    systems_df["system_id"] = [f"ADS_{i+1:03d}" for i in range(len(systems_df))]

    # ── Conflict scenarios ────────────────────
    scenarios_df = generate_conflict_scenarios(systems_df, countries_df, n_scenarios=700)

    # ── Save ─────────────────────────────────
# Save
    countries_path = os.path.join(DATA_DIR, "countries_profiles.csv")
    systems_path = os.path.join(DATA_DIR, "air_systems_enhanced.csv")
    scenarios_path = os.path.join(DATA_DIR, "conflict_scenarios.csv")

    countries_df.to_csv(countries_path, index=False)
    systems_df.to_csv(systems_path, index=False)
    scenarios_df.to_csv(scenarios_path, index=False)

    # ── Summary ──────────────────────────────
    print("=" * 65)
    print("   DRDO AIR DEFENCE PROJECT — DATA GENERATION COMPLETE")
    print("=" * 65)
    print(f"\n✅ Countries Profile   → {countries_path}")
    print(f"   Rows: {len(countries_df)} | Zones: "
          f"Red={len(countries_df[countries_df['risk_zone']=='Red'])}, "
          f"Yellow={len(countries_df[countries_df['risk_zone']=='Yellow'])}, "
          f"Green={len(countries_df[countries_df['risk_zone']=='Green'])}")

    print(f"\n✅ Air Systems         → {systems_path}")
    print(f"   Rows: {len(systems_df)} | "
          f"Modern={len(systems_df[systems_df['classification']=='Modern'])} | "
          f"Traditional={len(systems_df[systems_df['classification']=='Traditional'])}")
    print("   Systems per country:")
    for c, cnt in systems_df["country"].value_counts().items():
        print(f"      {c:<20} {cnt}")

    print(f"\n✅ Conflict Scenarios  → {scenarios_path}")
    print(f"   Rows: {len(scenarios_df)}")
    print(f"   Outcome distribution:\n"
          f"{scenarios_df['outcome'].value_counts().to_string()}")

    print(f"\n📊 Columns in Systems dataset ({len(systems_df.columns)}):")
    print(f"   {list(systems_df.columns)}")
    print("\n" + "=" * 65)
    print("   Step 1 COMPLETE → proceed to step2_train_models.py")
    print("=" * 65)
