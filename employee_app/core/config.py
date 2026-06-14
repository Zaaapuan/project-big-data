"""Shared paths, feature names, labels, and model constants."""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT_DIR / "employee_app"
UI_DIR = PACKAGE_DIR / "ui"
DATASET_PATH = ROOT_DIR / "data" / "employee_attrition.csv"
ARTIFACT_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "employee_profile_model.joblib"

# Incrementing this value invalidates artifacts created by an older pipeline.
PIPELINE_VERSION = "2.2.0"
RANDOM_STATE = 42
N_CLUSTERS = 3

NUMERIC_FEATURES = ["Age", "TotalWorkingYears", "Education"]
CATEGORICAL_FEATURES = ["Department"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

DEPARTMENTS = (
    "Human Resources",
    "Research & Development",
    "Sales",
)

EDUCATION_LEVELS = {
    1: "Below College",
    2: "College",
    3: "Bachelor",
    4: "Master",
    5: "Doctor",
}

CATEGORY_METADATA = {
    "Emerging Talent": {
        "color": "violet",
        "description": (
            "Kelompok early-career dengan usia lebih muda dan pengalaman "
            "kerja yang masih berkembang."
        ),
    },
    "Academic Achiever": {
        "color": "pink",
        "description": (
            "Kelompok dengan latar pendidikan lebih tinggi dibanding profil "
            "early-career lainnya."
        ),
    },
    "Seasoned Veteran": {
        "color": "ink",
        "description": (
            "Kelompok senior dengan rata-rata usia dan pengalaman kerja "
            "tertinggi di dataset."
        ),
    },
}
