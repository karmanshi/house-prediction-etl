"""
=============================================================
REIOS - Feature Engineering Tests
=============================================================

Purpose
-------
Validate feature engineering output before ML training.

Tests:
------
1. Dataset exists
2. Dataset loads correctly
3. Required columns exist
4. Dataset is not empty

=============================================================
"""


from pathlib import Path
import numpy as np
import pandas as pd
import pytest


##############################################################
# PROJECT PATHS
##############################################################

ROOT_DIR = Path(__file__).resolve().parent.parent


##############################################################
# REQUIRED COLUMNS
##############################################################

RAW_REQUIRED_COLUMNS = [

    "Land_Area",

    "Floor_Area",

    "Num_rooms",

    "Num_bathrooms",

    "Maintenance_Fees",

    "Latitude",

    "Longitude",

    "Crimerate",

    "dist_MRT",

    "dist_Hospital",

    "dist_School",

    "dist_BusStand",

    "dist_Airport",

    "Price",

]


ENGINEERED_REQUIRED_COLUMNS = [

    "price_log",

    "price_per_sqft",

    "size_ratio",

    "rooms_per_sqft",

    "bath_room_ratio",

    "amenity_count",

    "accessibility",

    "crime_access_interaction",

    "loc_price_median",

    "prop_type_price_median",

]


##############################################################
# TEST DATASET LOADING
##############################################################

def test_engineered_dataset_exists(
    engineered_df,
):
    """
    Verify feature engineered dataset exists
    and loads successfully.
    """

    assert isinstance(

        engineered_df,

        pd.DataFrame

    )


    assert len(engineered_df) > 0



##############################################################
# TEST DATASET SHAPE
##############################################################

def test_engineered_dataset_shape(
    engineered_df,
):
    """
    Dataset should contain reasonable amount
    of rows and columns.
    """

    rows, cols = engineered_df.shape


    assert rows > 0


    assert cols > 0



##############################################################
# TEST REQUIRED RAW FEATURES
##############################################################

def test_required_raw_features_exist(
    engineered_df,
):
    """
    Check original property features
    survived feature engineering.
    """

    missing = [

        col

        for col in RAW_REQUIRED_COLUMNS

        if col not in engineered_df.columns

    ]


    assert missing == [], (

        f"Missing raw columns: {missing}"

    )



##############################################################
# TEST ENGINEERED FEATURES
##############################################################

def test_engineered_features_exist(
    engineered_df,
):
    """
    Verify important engineered features
    are generated.
    """

    missing = [

        col

        for col in ENGINEERED_REQUIRED_COLUMNS

        if col not in engineered_df.columns

    ]


    assert missing == [], (

        f"Missing engineered features: {missing}"

    )
    
    
 ##############################################################
# TEST DUPLICATE ROWS
##############################################################

def test_no_excessive_duplicate_rows(
    engineered_df,
):
    """
    Feature engineering should not create
    unexpected duplicate properties.

    Small duplicates may exist naturally,
    therefore we check percentage.
    """

    duplicate_ratio = (

        engineered_df

        .duplicated()

        .mean()

    )


    assert duplicate_ratio < 0.05, (

        f"Too many duplicate rows: "
        f"{duplicate_ratio:.2%}"

    )



##############################################################
# TEST MISSING VALUES
##############################################################

def test_missing_values_in_model_features(
    engineered_df,
    feature_manager,
):
    """
    Model features should not contain
    excessive missing values.
    """

    all_features = (

        feature_manager

        .get_all_model_features()

    )


    existing_features = [

        col

        for col in all_features

        if col in engineered_df.columns

    ]


    missing_percentage = (

        engineered_df[existing_features]

        .isnull()

        .mean()

    )


    problematic = (

        missing_percentage

        [missing_percentage > 0.05]

        .index

        .tolist()

    )


    assert problematic == [], (

        f"Features with high missing values: "
        f"{problematic}"

    )



##############################################################
# TEST NUMERIC FEATURES
##############################################################

def test_numeric_features_are_numeric(
    engineered_df,
    feature_manager,
):
    """
    ML models require numerical input.

    Checks that numerical features
    are not accidentally stored as strings.
    """

    numeric_features = (

        feature_manager

        .get_numeric_features(engineered_df)

    )


    existing = [

        col

        for col in numeric_features

        if col in engineered_df.columns

    ]


    non_numeric = []


    for col in existing:

        if not pd.api.types.is_numeric_dtype(

            engineered_df[col]

        ):

            non_numeric.append(col)



    assert non_numeric == [], (

        f"Non numeric features found: "
        f"{non_numeric}"

    )



##############################################################
# TEST TARGET VARIABLE
##############################################################

def test_target_variable(
    engineered_df,
):
    """
    Validate regression target.

    Hedonic model predicts:
        price_log
    """

    assert "price_log" in engineered_df.columns


    assert (

        engineered_df["price_log"]

        .notnull()

        .all()

    )


    assert (

        (engineered_df["price_log"] >= 0)

        .all()

    )



##############################################################
# TEST FEATURE MANAGER CONSISTENCY
##############################################################

def test_feature_manager_features_exist(
    engineered_df,
    feature_manager,
):
    """
    Ensure FeatureManager lists only
    existing dataframe columns.
    """

    model_features = (

        feature_manager

        .get_all_model_features()

    )


    missing = [

        feature

        for feature in model_features

        if feature not in engineered_df.columns

    ]


    assert missing == [], (

        "FeatureManager contains missing "
        f"columns: {missing}"

    )   
    
    
##############################################################
# TEST INFINITE VALUES
##############################################################

def test_no_infinite_values(
    engineered_df,
):
    """
    ML models cannot handle infinite values.

    Check:
    inf
    -inf
    """

    numeric_df = (

        engineered_df

        .select_dtypes(

            include=["number"]

        )

    )


    infinite_values = (

        np.isinf(numeric_df)

        .sum()

        .sum()

    )


    assert infinite_values == 0, (

        f"Found {infinite_values} infinite values"

    )



##############################################################
# TEST PRICE FEATURE RANGE
##############################################################

def test_price_values_valid(
    engineered_df,
):
    """
    Property price should be positive.

    Negative or zero prices indicate
    data processing problems.
    """

    assert "Price" in engineered_df.columns


    assert (

        engineered_df["Price"]

        > 0

    ).all()



##############################################################
# TEST PRICE LOG TRANSFORMATION
##############################################################

def test_price_log_transformation(
    engineered_df,
):
    """
    Verify:

    price_log = log1p(Price)

    """

    sample = engineered_df.sample(

        min(100, len(engineered_df)),

        random_state=42

    )


    expected = np.log1p(

        sample["Price"]

    )


    difference = (

        abs(

            sample["price_log"]

            -

            expected

        )

    )


    assert (

        difference.max()

        < 0.001

    ), "price_log transformation incorrect"



##############################################################
# TEST PRICE PER SQFT
##############################################################

def test_price_per_sqft_formula(
    engineered_df,
):
    """
    Verify:

    price_per_sqft =
        Price / Floor_Area
    """

    sample = engineered_df.sample(

        min(100, len(engineered_df)),

        random_state=42

    )


    expected = (

        sample["Price"]

        /

        sample["Floor_Area"]

    )


    difference = (

        abs(

            sample["price_per_sqft"]

            -

            expected

        )

    )


    assert (

        difference.mean()

        < 0.01

    ), "price_per_sqft calculation incorrect"



##############################################################
# TEST ROOMS PER SQFT
##############################################################

def test_rooms_per_sqft_formula(
    engineered_df,
):
    """
    Verify:

    rooms_per_sqft =
        Num_rooms / Floor_Area
    """

    if "rooms_per_sqft" not in engineered_df.columns:

        pytest.skip(
            "rooms_per_sqft not generated"
        )


    sample = engineered_df.sample(

        min(100, len(engineered_df)),

        random_state=42

    )


    expected = (

        sample["Num_rooms"]

        /

        sample["Floor_Area"]

    )


    difference = (

        abs(

            sample["rooms_per_sqft"]

            -

            expected

        )

    )


    assert (

        difference.mean()

        < 0.001

    )



##############################################################
# TEST BATHROOM RATIO
##############################################################

def test_bathroom_ratio_formula(
    engineered_df,
):
    """
    Verify:

    bath_room_ratio =
        Num_bathrooms / Num_rooms
    """

    if "bath_room_ratio" not in engineered_df.columns:

        pytest.skip(
            "bath_room_ratio not generated"
        )


    sample = engineered_df.sample(

        min(100, len(engineered_df)),

        random_state=42

    )


    expected = (

        sample["Num_bathrooms"]

        /

        sample["Num_rooms"]

    )


    difference = (

        abs(

            sample["bath_room_ratio"]

            -

            expected

        )

    )


    assert (

        difference.mean()

        < 0.001

    )
    