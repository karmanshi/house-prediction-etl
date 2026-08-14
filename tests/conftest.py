"""
=============================================================
REIOS - Pytest Shared Fixtures
=============================================================

Purpose
-------
Shared pytest fixtures used across all test modules.

Provides
--------
✓ Project paths
✓ Engineered dataset
✓ FeatureManager
✓ Random seed
✓ Common reusable objects

Author
------
Bhavya

=============================================================
"""

from __future__ import annotations

##############################################################
# STANDARD LIBRARIES
##############################################################

from pathlib import Path
import random

##############################################################
# THIRD PARTY
##############################################################

import numpy as np
import pandas as pd
import pytest

##############################################################
# PROJECT MODULES
##############################################################

from training.feature_manager import FeatureManager

##############################################################
# PROJECT PATHS
##############################################################

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"

RAW_DIR = DATA_DIR / "raw"

PROCESSED_DIR = DATA_DIR / "processed"

MODEL_DIR = ROOT_DIR / "models"

CONFIG_DIR = ROOT_DIR / "configs"

##############################################################
# RANDOM SEED
##############################################################

@pytest.fixture(scope="session")
def random_seed():
    """
    Shared random seed.

    Keeps all tests deterministic.
    """

    seed = 42

    np.random.seed(seed)

    random.seed(seed)

    return seed


##############################################################
# FEATURE MANAGER
##############################################################

@pytest.fixture(scope="session")
def feature_manager():
    """
    Shared FeatureManager instance.
    """

    return FeatureManager()


##############################################################
# ENGINEERED DATASET
##############################################################

@pytest.fixture(scope="session")
def engineered_df():
    """
    Load feature engineered dataset once.

    Returns
    -------
    pandas.DataFrame
    """

    dataset = PROCESSED_DIR / "feature_engineered.csv"

    assert dataset.exists(), (

        f"Dataset not found:\n{dataset}"

    )

    df = pd.read_csv(dataset)

    return df


##############################################################
# SCORED DATASET
##############################################################

@pytest.fixture(scope="session")
def scored_df():
    """
    Load scored dataset.

    Used after scorer.py has been executed.
    """

    dataset = PROCESSED_DIR / "scored_properties.csv"

    assert dataset.exists(), (

        f"Scored dataset not found:\n{dataset}"

    )

    return pd.read_csv(dataset)

##############################################################
# MODEL LOADING
##############################################################

import joblib

##############################################################
# HEDONIC MODEL
##############################################################

@pytest.fixture(scope="session")
def hedonic_model():
    """
    Load trained LightGBM Hedonic model.
    """

    model_path = MODEL_DIR / "lgbm_hedonic.pkl"

    assert model_path.exists(), (

        f"Hedonic model not found:\n{model_path}"

    )

    return joblib.load(model_path)


##############################################################
# ISOLATION FOREST
##############################################################

@pytest.fixture(scope="session")
def anomaly_model():
    """
    Load trained Isolation Forest.
    """

    model_path = MODEL_DIR / "iso_forest.pkl"

    assert model_path.exists(), (

        f"Isolation Forest model not found:\n{model_path}"

    )

    return joblib.load(model_path)


##############################################################
# CLASSIFIER MODEL
##############################################################

@pytest.fixture(scope="session")
def classifier_model():
    """
    Load trained Tier Classifier.
    """

    model_path = MODEL_DIR / "lgbm_classifier.pkl"

    assert model_path.exists(), (

        f"Classifier model not found:\n{model_path}"

    )

    return joblib.load(model_path)


##############################################################
# TIER LABEL ENCODER
##############################################################

@pytest.fixture(scope="session")
def tier_encoder():
    """
    Load LabelEncoder used for tiers.
    """

    encoder_path = MODEL_DIR / "tier_encoder.pkl"

    assert encoder_path.exists(), (

        f"Tier encoder not found:\n{encoder_path}"

    )

    return joblib.load(encoder_path)




##############################################################
# SAMPLE ENGINEERED ROW
##############################################################

@pytest.fixture(scope="session")
def sample_engineered_row(
    engineered_df,
):
    """
    Return one engineered property.

    Used by almost every test.
    """

    return engineered_df.iloc[0].copy()


##############################################################
# HEDONIC FEATURES
##############################################################

@pytest.fixture(scope="session")
def hedonic_features(
    feature_manager,
    engineered_df,
):
    """
    Hedonic model features.
    """

    return feature_manager.get_hedonic_features(engineered_df)


##############################################################
# ANOMALY FEATURES
##############################################################

@pytest.fixture(scope="session")
def anomaly_features(
    feature_manager,
    engineered_df,
):
    """
    Isolation Forest features.
    """

    return feature_manager.get_anomaly_features(engineered_df)


##############################################################
# CLASSIFIER FEATURES
##############################################################

@pytest.fixture(scope="session")
def classifier_features(
    feature_manager,
    engineered_df,
):
    """
    Tier classifier features.
    """

    return feature_manager.get_classifier_features(engineered_df)


##############################################################
# SAMPLE HEDONIC INPUT
##############################################################

@pytest.fixture(scope="session")
def sample_hedonic_input(
    engineered_df,
    hedonic_features,
):
    """
    One-row dataframe for regression model.
    """

    return (

        engineered_df

        [hedonic_features]

        .fillna(0)

        .iloc[[0]]

    )


##############################################################
# SAMPLE ANOMALY INPUT
##############################################################

@pytest.fixture(scope="session")
def sample_anomaly_input(
    engineered_df,
    anomaly_features,
):
    """
    One-row dataframe for Isolation Forest.
    """

    return (

        engineered_df

        [anomaly_features]

        .fillna(0)

        .iloc[[0]]

    )


##############################################################
# SAMPLE CLASSIFIER INPUT
##############################################################

@pytest.fixture(scope="session")
def sample_classifier_input(
    engineered_df,
    classifier_features,
):
    """
    One-row dataframe for classifier.
    """

    cols = [

        c

        for c in classifier_features

        if c in engineered_df.columns

    ]

    return (

        engineered_df

        [cols]

        .fillna(0)

        .iloc[[0]]

    )
    
##############################################################
# TEMPORARY DIRECTORY
##############################################################

@pytest.fixture
def temp_output_dir(
    tmp_path,
):
    """
    Creates temporary directory for testing.

    pytest automatically deletes it after test completion.

    Used for:
    - testing model saving
    - testing csv generation
    - testing artifact creation
    """

    return tmp_path


##############################################################
# SAMPLE MODEL OUTPUT
##############################################################

@pytest.fixture(scope="session")
def sample_model_output():
    """
    Example prediction output.

    Used for testing:
    - scoring
    - API responses
    - schema validation
    """

    return {

        "predicted_price": 850000,

        "residual_pct": -12.5,

        "anomaly_score": -0.18,

        "anomaly_label": 1,

        "value_gap_score": 0.82,

        "growth_score": 0.65,

        "access_score": 0.75,

        "safety_score": 0.90,

        "anomaly_score_n": 0.70,

        "opportunity_score": 78.5,

        "tier": "Excellent",

        "percentile": 94.2,

    }


##############################################################
# SAMPLE API PAYLOAD
##############################################################

@pytest.fixture(scope="session")
def sample_property_payload():
    """
    Sample property input.

    Used by test_api.py

    This represents a user sending
    a new property for prediction.
    """

    return {

        "Land_Area": 1200,

        "Floor_Area": 950,

        "Num_rooms": 3,

        "Num_bathrooms": 2,

        "Maintenance_Fees": 2500,

        "Latitude": 1.3521,

        "Longitude": 103.8198,

        "Crimerate": 0.03,

        "dist_MRT": 500,

        "dist_Hospital": 1500,

        "dist_School": 700,

        "dist_BusStand": 300,

        "dist_Airport": 12000,

        "Property_Type": "Apartment",

        "Condition": "New",

        "Furnishing_Status": "Fully Furnished",

    }


##############################################################
# ASSERT REQUIRED MODEL FILES
##############################################################

@pytest.fixture(scope="session")
def required_model_files():

    return [

        MODEL_DIR / "lgbm_hedonic.pkl",

        MODEL_DIR / "iso_forest.pkl",

        MODEL_DIR / "lgbm_classifier.pkl",

        MODEL_DIR / "tier_encoder.pkl",

        # MODEL_DIR / "label_encoders.pkl",

    ]

import pytest
from training.scorer import OpportunityScorer


@pytest.fixture(scope="session")
def scorer():

    s = OpportunityScorer()

    s.load_models(include_classifier=False)

    return s
