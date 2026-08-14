"""
=============================================================
REIOS Feature Manager
=============================================================

Purpose
-------
Centralized feature definitions for all machine learning
models used in REIOS.

Models
------
1. Hedonic Price Prediction
2. Isolation Forest
3. Tier Classification

This file prevents duplicated feature lists across the
training pipeline.

Author
------
Bhavya
=============================================================
"""

from __future__ import annotations

import joblib
from pathlib import Path


from typing import List

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent

FEATURE_METADATA_DIR = ROOT_DIR / "feature_metadata"

###############################################################
# FEATURE MANAGER
###############################################################


class FeatureManager:
    """
    Central feature repository.

    Every model imports features from here.

    Benefits
    --------
    ✔ Single source of truth

    ✔ Easier maintenance

    ✔ Prevents inconsistent feature lists

    ✔ Backend uses same features as training
    """

    ###########################################################
    # TARGET
    ###########################################################

    TARGET = "price_log"

    ###########################################################
    # NUMERIC FEATURES
    ###########################################################

    NUMERIC_FEATURES = [

        "Land_Area",

        "Floor_Area",

        "Num_rooms",

        "Num_bathrooms",

        "Maintenance_Fees",

        "Latitude",

        "Longitude",

        "dist_MRT",

        "dist_Hospital",

        "dist_School",

        "dist_BusStand",

        "dist_Airport",

        "Crimerate",

        "size_ratio",

        "rooms_per_sqft",

        "bath_room_ratio",

        "amenity_count",

        "accessibility",

        "crime_access_interaction",

        "loc_price_median",

        "prop_type_price_median",

    ]

    ###########################################################
    # AMENITIES
    ###########################################################

    AMENITY_FEATURES = [

        "Bar",

        "Elevator",

        "Garden",

        "Gym",

        "Parking",

        "Swimming_Pool",

        "WiFi",

        #"Air_Conditioning",

    ]

    ###########################################################
    # ONE HOT PREFIXES
    ###########################################################

    OHE_PREFIXES = [

        "Condition_",

        "Location_",

        "Property_Type_",

        "Kitchen_Type_",

        "View_",

        "Furnishing_Status_",
        
        "Heating_",

        "Balcony_",

    ]

    ###########################################################
    # TARGET LEAKAGE
    ###########################################################

    LEAKAGE_COLUMNS = [

        "Price",

        "price_log",

        "price_per_sqft",

        "predicted_price",

        "residual_pct",

        "opportunity_score",

        "tier",

        "percentile",

    ]

    ###########################################################
    # CLASSIFIER EXTRA FEATURES
    ###########################################################

    CLASSIFIER_EXTRA = [

        "residual_feat",

        "anomaly_feat",

        "value_gap_score",

        "growth_score",

        "access_score",

        "safety_score",

        "anomaly_score_n",

    ]
    ###########################################################
# FEATURE DETECTION UTILITIES
###########################################################

    @classmethod
    def get_numeric_features(
        cls,
        df: pd.DataFrame,
    ) -> List[str]:
        """
        Return available numeric features.

        Parameters
        ----------
        df : DataFrame

        Returns
        -------
        List[str]
        """

        return [

            col

            for col in cls.NUMERIC_FEATURES

            if col in df.columns

        ]


    ###########################################################

    @classmethod
    def get_amenity_features(
        cls,
        df: pd.DataFrame,
    ) -> List[str]:
        """
        Return available amenity features.
        """

        return [

            col

            for col in cls.AMENITY_FEATURES

            if col in df.columns

        ]


    ###########################################################

    @classmethod
    def get_ohe_features(
        cls,
        df: pd.DataFrame,
    ) -> List[str]:
        """
        Automatically detect one-hot encoded columns.

        Example

        Location_CBD

        Property_Type_Apartment

        View_River
        """

        features = []

        for column in df.columns:

            if any(

                column.startswith(prefix)

                for prefix in cls.OHE_PREFIXES

            ):

                features.append(column)

        return sorted(features)


    ###########################################################
    # HEDONIC FEATURES
    ###########################################################

    @classmethod
    def get_hedonic_features(
        cls,
        df: pd.DataFrame,
    ) -> List[str]:
        """
        Return complete feature list for
        Hedonic Regression.
        """

        features = (

            cls.get_numeric_features(df)

            +

            cls.get_amenity_features(df)

            +

            cls.get_ohe_features(df)

        )

        #######################################################
        # REMOVE TARGET LEAKAGE
        #######################################################

        features = [

            col

            for col in features

            if col not in cls.LEAKAGE_COLUMNS

        ]

        #######################################################
        # REMOVE DUPLICATES
        #######################################################

        features = list(

            dict.fromkeys(features)

        )

        return features


    ###########################################################
    # ANOMALY FEATURES
    ###########################################################

    @classmethod
    def get_anomaly_features(
        cls,
        df: pd.DataFrame,
    ) -> List[str]:
        """
        Return feature list for
        Isolation Forest.
        """

        candidate_features = [

            "price_per_sqft",

            "Floor_Area",

            "Num_rooms",

            "Num_bathrooms",

            "Maintenance_Fees",

            "Latitude",

            "Longitude",

            "dist_MRT",

            "Crimerate",

        ]

        features = [

            feature

            for feature in candidate_features

            if feature in df.columns

        ]

        # automatically include all one-hot columns

        features += cls.get_ohe_features(df)

        return list(dict.fromkeys(features))


    ###########################################################
    # CLASSIFIER FEATURES
    ###########################################################

    @classmethod
    def get_classifier_features(
        cls,
        df: pd.DataFrame,
    ) -> List[str]:
        """
        Return features for Tier Classifier.
        """

        features = (

            cls.get_hedonic_features(df)

            +

            [

                feature

                for feature in cls.CLASSIFIER_EXTRA

                if feature in df.columns

            ]

        )

        features = list(

            dict.fromkeys(features)

        )

        return features

    ###########################################################
    # FEATURE VALIDATION
    ###########################################################

    @classmethod
    def validate_features(
        cls,
        df: pd.DataFrame,
        feature_list: List[str],
        raise_error: bool = True,
    ) -> List[str]:
        """
        Validate feature availability.

        Parameters
        ----------
        df : DataFrame

        feature_list : List[str]

        raise_error : bool

        Returns
        -------
        missing_features
        """

        missing = [

            feature

            for feature in feature_list

            if feature not in df.columns

        ]

        if missing and raise_error:

            raise ValueError(

                "Missing Features:\n"

                + "\n".join(missing)

            )

        return missing


    ###########################################################
    # CLEAN FEATURE MATRIX
    ###########################################################

    @classmethod
    def clean_features(
        cls,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Standard feature cleaning.

        • Replace inf
        • Replace -inf
        • Fill NaN
        """

        X = X.copy()

        X = X.replace(

            [

                np.inf,

                -np.inf,

            ],

            np.nan,

        )

        X = X.fillna(0)

        return X


    ###########################################################
    # HEDONIC DATASET
    ###########################################################

    @classmethod
    def prepare_hedonic_data(
        cls,
        df: pd.DataFrame,
    ):
        """
        Prepare Hedonic Regression data.

        Returns
        -------

        X

        y

        features
        """

        #if "size_ratio" not in df.columns:

            #df = cls.prepare_prediction_data(df)
        if "size_ratio" not in df.columns and "loc_price_median" not in df.columns:
            df = cls.prepare_prediction_data(df)
        
        
        features = cls.get_hedonic_features(df)

        cls.validate_features(

            df,

            features,

        )

        X = cls.clean_features(

            df[features]

        )

        if cls.TARGET in df.columns:
            y = df[cls.TARGET]
        else:
            y = None

        return X, y, features


    ###########################################################
    # ANOMALY DATASET
    ###########################################################

    @classmethod
    def prepare_anomaly_data(
        cls,
        df: pd.DataFrame,
    ):
        """
        Prepare Isolation Forest dataset.
        """

        #if "size_ratio" not in df.columns:

           # df = cls.prepare_prediction_data(df)
        
        if "size_ratio" not in df.columns and "loc_price_median" not in df.columns:
            df = cls.prepare_prediction_data(df)
        
        
        
        features = cls.get_anomaly_features(df)

        cls.validate_features(

            df,

            features,

        )

        X = cls.clean_features(

            df[features]

        )

        return X, features


    ###########################################################
    # CLASSIFIER DATASET
    ###########################################################

    @classmethod
    def prepare_classifier_data(
        cls,
        df: pd.DataFrame,
        target_column: str = "tier",
    ):
        """
        Prepare classifier dataset.

        """

        
        if "size_ratio" not in df.columns and "loc_price_median" not in df.columns:
            df = cls.prepare_prediction_data(df)
        
        features = cls.get_classifier_features(df)

        cls.validate_features(

            df,

            features,

        )

        X = cls.clean_features(

            df[features]

        )

        # y = df[target_column]
        if target_column in df.columns:
            y = df[target_column]
        else:
            y = None


        return X, y, features


    ###########################################################
    # FEATURE REPORT
    ###########################################################

    @classmethod
    def feature_report(
        cls,
        df: pd.DataFrame,
    ):
        """
        Generate feature summary.
        """

        hedonic = cls.get_hedonic_features(df)

        anomaly = cls.get_anomaly_features(df)

        classifier = cls.get_classifier_features(df)

        report = {

            "rows":

                len(df),

            "columns":

                len(df.columns),

            "hedonic_features":

                len(hedonic),

            "anomaly_features":

                len(anomaly),

            "classifier_features":

                len(classifier),

        }

        return report


    ###########################################################
    # PRINT REPORT
    ###########################################################

    @classmethod
    def print_report(
        cls,
        df: pd.DataFrame,
    ):
        """
        Print feature summary.
        """

        report = cls.feature_report(df)

        print("\n")

        print("=" * 60)

        print("FEATURE REPORT")

        print("=" * 60)

        for key, value in report.items():

            print(

                f"{key:25} : {value}"

            )

        print("=" * 60)
        
    @classmethod
    def get_all_model_features(cls, df):
        """
        Returns all unique features used by every model.
        """
        hedonic = cls.get_hedonic_features(df)

        anomaly = cls.get_anomaly_features(df)

        classifier = cls.get_classifier_features(df)

        return sorted(

            set(

                hedonic

                + anomaly

                + classifier

            )

        ) 
        
        #FOR BACKEND 
    @staticmethod
    def load_feature_metadata():

        metadata = {}

        metadata["feature_columns"] = joblib.load(
            FEATURE_METADATA_DIR / "feature_columns.pkl"
        )

        metadata["feature_order"] = joblib.load(
            FEATURE_METADATA_DIR / "feature_order.pkl"
        )

        metadata["numeric_columns"] = joblib.load(
            FEATURE_METADATA_DIR / "numeric_columns.pkl"
        )

        metadata["categorical_levels"] = joblib.load(
            FEATURE_METADATA_DIR / "categorical_levels.pkl"
        )

        metadata["location_price"] = joblib.load(
            FEATURE_METADATA_DIR / "location_price.pkl"
        )

        metadata["property_price"] = joblib.load(
            FEATURE_METADATA_DIR / "property_price.pkl"
        )

        metadata["training_medians"] = joblib.load(
            FEATURE_METADATA_DIR / "training_medians.pkl"
        )

        metadata["training_modes"] = joblib.load(
            FEATURE_METADATA_DIR / "training_modes.pkl"
        )
        
        metadata["normalization_stats"] = joblib.load(

            FEATURE_METADATA_DIR/

            "normalization_stats.pkl"

        )

        return metadata
    #FOR BACKEND
    @staticmethod
    def prepare_prediction_data(df):

        meta = FeatureManager.load_feature_metadata()

        df = df.copy()
        raw_df = df.copy()
        # if "Swimming Pool" in df.columns:
        #     df.rename(
        #         columns={
        #             "Swimming Pool": "Swimming_Pool"
        #         },
        #         inplace=True
        #     )

        ##########################################################
        # Derived Features
        ##########################################################

        if "Price" in df.columns:
            df["price_log"] = np.log1p(df["Price"])
        df["size_ratio"] = (
            df["Floor_Area"] /
            df["Land_Area"].replace(0,1)
        )

        df["rooms_per_sqft"] = (
            df["Num_rooms"] /
            df["Floor_Area"].replace(0,1)
        )

        df["bath_room_ratio"] = (
            df["Num_bathrooms"] /
            df["Num_rooms"].replace(0,1)
        )

        df["accessibility"] = (

            df["dist_MRT"]

            +

            df["dist_BusStand"]

        ) / 2

        amenity_cols = [

            "Bar",

            "Elevator",

            "Garden",

            "Gym",

            "Parking",

            "Swimming_Pool",

            "WiFi",

            #"Air_Conditioning",

        ]
        
        for col in amenity_cols:

            if col not in df.columns:

                df[col] = 0
        
        

        df["amenity_count"] = df[amenity_cols].sum(axis=1)

        df["crime_access_interaction"] = (

            df["Crimerate"]

            *

            df["accessibility"]

        )

        ##########################################################
        # Median Encodings
        ##########################################################

        global_loc = np.mean(
            list(meta["location_price"].values())
        )
        # Reconstruct Location from one-hot encoded columns
        # location_cols = [
        #     col for col in df.columns 
        #     if col.startswith("Location_")
        # ]

        # if not location_cols:
        #     raise ValueError("No Location columns found.")

        # df["Location"] = (
        #     df[location_cols]
        #     .idxmax(axis=1)
        #     .str.replace("Location_", "", regex=False)
        # )
            
        # if df[location_cols].sum(axis=1).eq(0).any():
        #     raise ValueError("No Location columns found in dataframe")
        if "Location" not in df.columns:
            raise ValueError("Location column missing.")

        df["loc_price_median"] = (
            df["Location"]
            .map(meta["location_price"])
            .fillna(global_loc)
        )

        global_prop = np.mean(
            list(meta["property_price"].values())
        )

        # Reconstruct Property_Type from one-hot encoded columns
        # property_cols = [
        #     col for col in df.columns
        #     if col.startswith("Property_Type_")
        # ]

       
        # if not property_cols:
        #     raise ValueError("No Property_Type columns found.")

        # df["Property_Type"] = (
        #     df[property_cols]
        #     .idxmax(axis=1)
        #     .str.replace("Property_Type_", "", regex=False)
        # )
        if "Property_Type" not in df.columns:
            raise ValueError("Property_Type column missing.")
        
        # if df[location_cols].sum(axis=1).eq(0).any():
        #             raise ValueError("No Location columns found in dataframe")
           
        

        df["prop_type_price_median"] = (

            df["Property_Type"]

            .map(meta["property_price"])

            .fillna(global_prop)

        )

        ##########################################################
        # Distance from Centre
        ##########################################################

        lat_mean = meta["training_medians"]["Latitude"]

        lon_mean = meta["training_medians"]["Longitude"]

        df["dist_from_centre"] = np.sqrt(

            (df["Latitude"]-lat_mean)**2 +

            (df["Longitude"]-lon_mean)**2

        )

        ##########################################################
        # One Hot Encoding
        ##########################################################

        categorical = [

            "Property_Type",

            "Condition",

            "View",

            "Kitchen_Type",

            "Location",

            "Furnishing_Status",


        ]

        existing_columns = [
            col for col in categorical
            if col in df.columns
        ]

        if existing_columns:
            df = pd.get_dummies(
                df,
                columns=existing_columns,
                drop_first=False
            )

        ##########################################################
        # Create Engineered Feature Matrix
        ##########################################################

        engineered = df.copy()
        # Normalize feature names
        engineered.columns = (
            engineered.columns
            .str.replace(" ", "_")
        )


        # Remove duplicate columns
        engineered = engineered.loc[
            :,
            ~engineered.columns.duplicated()
        ]




        ##########################################################
        # Add Missing Columns
        ##########################################################

        
        
        
        for col in meta["feature_order"]:

            if col not in engineered.columns:

                engineered[col] = 0

        ##########################################################
        # Keep Training Feature Order
        ##########################################################

        engineered = engineered[
            meta["feature_order"]
        ]

        ##########################################################
        # Fill Missing Values
        ##########################################################

        for col, val in meta["training_medians"].items():

            if col in engineered.columns:

                engineered[col] = engineered[col].fillna(val)

        ##########################################################
        # Return Engineered Features
        ##########################################################

        # Restore required business columns

        for col in [
            "Location",
            "Property_Type",
            "Crimerate",
            "accessibility",
            "loc_price_median",
            "prop_type_price_median"
        ]:
            if col in df.columns:
                engineered[col] = df[col]


        return engineered








