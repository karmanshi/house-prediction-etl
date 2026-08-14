"""
=============================================================
REIOS - Composite Opportunity Scorer
=============================================================

Purpose
-------
Generate investment scores for properties using

1. Hedonic Price Model
2. Isolation Forest
3. Composite Scoring Engine
4. Tier Classifier

This module NEVER trains models.

It only loads trained models and performs inference.

Author
------
Bhavya

=============================================================
"""

from __future__ import annotations

##############################################################
# STANDARD LIBRARIES
##############################################################

import logging
from pathlib import Path
from typing import Dict

##############################################################
# THIRD PARTY
##############################################################

import joblib
import numpy as np
import pandas as pd
import yaml

from scipy.stats import percentileofscore

##############################################################
# PROJECT MODULES
##############################################################

from training.feature_manager import FeatureManager

##############################################################
# PROJECT PATHS
##############################################################

ROOT_DIR = Path(__file__).resolve().parent.parent


CONFIG_DIR = ROOT_DIR / "configs"

MODEL_DIR = ROOT_DIR / "models"

##############################################################
# LOGGING
##############################################################

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s",

)

logger = logging.getLogger(__name__)

##############################################################
# SCORER
##############################################################

class OpportunityScorer:
    """
    Production inference engine.

    Responsibilities

    ✔ Load trained models

    ✔ Predict property price

    ✔ Detect anomaly

    ✔ Compute investment score

    ✔ Predict tier

    ✔ Return results
    """

    ##########################################################
    # INITIALIZATION
    ##########################################################

    def __init__(self):

        logger.info("=" * 60)
        logger.info("Initializing Opportunity Scorer")
        logger.info("=" * 60)

        ######################################################
        # CONFIG
        ######################################################

        self.config = self._load_config()

        ######################################################
        # MODELS
        ######################################################

        self.hedonic_model = None

        self.anomaly_model = None

        self.classifier = None

        self.encoder = None
        
        self.metadata = FeatureManager.load_feature_metadata()
        
        self.norm_stats = self.metadata["normalization_stats"]

        ######################################################
        # WEIGHTS
        ######################################################

        self.weights = self._load_weights()

        logger.info("Initialization Complete")

    ##########################################################
    # LOAD CONFIG
    ##########################################################

    def _load_config(self) -> Dict:

        config_path = (

            CONFIG_DIR /

            "model_config.yaml"

        )

        if not config_path.exists():

            return {}

        with open(

            config_path,

            "r",

        ) as fp:

            return yaml.safe_load(fp)

    ##########################################################
    # LOAD WEIGHTS
    ##########################################################

    def _load_weights(self):

        defaults = {

            "value_gap_score":0.35,

            "growth_score":0.20,

            "access_score":0.15,

            "safety_score":0.12,

            "anomaly_score_n":0.05,

        }

        cfg = self.config.get(

            "scoring_weights",

            {}

        )

        if cfg:

            defaults = {

                "value_gap_score":cfg.get("value_gap",0.35),

                "growth_score":cfg.get("growth",0.20),

                "access_score":cfg.get("access",0.15),

                "safety_score":cfg.get("safety",0.12),

                "anomaly_score_n":cfg.get("anomaly",0.05),

            }

        return defaults
    
    ##########################################################
    # LOAD TRAINED MODELS
    ##########################################################

    def load_models(self,include_classifier=False):
        """
        Load all trained models required for inference.
        """
        import json
        logger.info("=" * 60)
        logger.info("Loading Trained Models")
        logger.info("=" * 60)

        self.hedonic_model = joblib.load(
            MODEL_DIR / "lgbm_hedonic.pkl"
        )

        self.anomaly_model = joblib.load(
            MODEL_DIR / "iso_forest.pkl"
        )
        self.anomaly_features = json.load(
            open(
                MODEL_DIR / "anomaly_features.json"
            )
        )
        if include_classifier:

            self.classifier = joblib.load(
                MODEL_DIR / "lgbm_classifier.pkl"
            )

            self.encoder = joblib.load(
                MODEL_DIR / "tier_encoder.pkl"
            )

        logger.info("All models loaded successfully.")

        return self


    ##########################################################
    # MIN-MAX CLIP NORMALIZATION
    ##########################################################

    @staticmethod
    def minmax_clip(

        series,

        p1,

        p99,

    ):

        clipped = series.clip(

            lower=p1,

            upper=p99,

        )

        return (

            clipped-p1

        )/(p99-p1+1e-9)


    ##########################################################
    # CALCULATE FACTOR SCORES
    ##########################################################

    def calculate_factor_scores(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute normalized investment factor scores.
        """

        logger.info("Calculating factor scores...")

        #######################################################
        # VALUE GAP
        #######################################################

        stats = self.norm_stats
        
        # -------------------------------------------------------
        # Safety: add missing normalization stats dynamically
        # -------------------------------------------------------
        required_stats = [
            "residual_pct",
            "loc_price_median",
            "accessibility",
            "Crimerate",
            "anomaly_score",
        ]


        for col in required_stats:

            if col not in stats:

                raise ValueError(
                    f"Missing normalization stats for {col}"
                )
        df["value_gap_score"] = self.minmax_clip(

            -df["residual_pct"],

            -stats["residual_pct"]["p99"],

            -stats["residual_pct"]["p1"]

        )

        #######################################################
        # LOCATION GROWTH
        #######################################################

        df["growth_score"]=self.minmax_clip(

            df["loc_price_median"],

            stats["loc_price_median"]["p1"],

            stats["loc_price_median"]["p99"]

        )

        #######################################################
        # ACCESSIBILITY
        #######################################################

        df["access_score"]=self.minmax_clip(

            -df["accessibility"],

            -stats["accessibility"]["p99"],

            -stats["accessibility"]["p1"]

        )

        #######################################################
        # SAFETY
        #######################################################

        df["safety_score"]=self.minmax_clip(

            -df["Crimerate"],

            -stats["Crimerate"]["p99"],

            -stats["Crimerate"]["p1"]

        )

        #######################################################
        # ANOMALY
        #######################################################

        df["anomaly_score_n"]=self.minmax_clip(

            -df["anomaly_score"],

            -stats["anomaly_score"]["p99"],

            -stats["anomaly_score"]["p1"]

        )

        return df


    ##########################################################
    # OPPORTUNITY SCORE
    ##########################################################

    def calculate_opportunity_score(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Weighted composite investment score.
        """

        logger.info("Calculating opportunity score...")

        score = np.zeros(len(df))

        for feature, weight in self.weights.items():

            score += df[feature] * weight

        total_weight = sum(self.weights.values())

        df["opportunity_score"] = (
            score / total_weight * 100
        ).clip(0, 100).round(1)

        return df


    ##########################################################
    # ASSIGN INVESTMENT TIER
    ##########################################################

    def assign_tier(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert score into investment tier.
        """

        thresholds = self.config.get(

            "tier_thresholds",

            {}

        )

        low = thresholds.get("low", 0)
        fair = thresholds.get("fair", 30)
        good = thresholds.get("good", 50)
        excellent = thresholds.get("excellent", 70)

        df["tier"] = pd.cut(

            df["opportunity_score"],

            bins=[

                low,
                fair,
                good,
                excellent,
                101,

            ],

            labels=[

                "Low",

                "Fair",

                "Good",

                "Excellent",

            ],

            include_lowest=True,

        ).astype(str)

        return df


    ##########################################################
    # CALCULATE PERCENTILE
    ##########################################################

    def calculate_percentile(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute percentile rank of every property.
        """

        scores = df["opportunity_score"].values

        df["percentile"] = (
            df["opportunity_score"]
            .rank(pct=True)
            *100
        )

        return df
        
    ##########################################################
    # PREPARE MODEL FEATURES
    ##########################################################

    def prepare_features(
        self,
        df: pd.DataFrame,
    ):
        """
        Prepare features required by each model.
        If raw property input is supplied,
        FeatureManager automatically performs
        feature engineering.
        """

        logger.info("Preparing model features...")

        
        
        
        
        
        # Hedonic Features
        
        X_hedonic, _, _ = FeatureManager.prepare_hedonic_data(df)

        # Anomaly Features
        
        X_anomaly, _ = FeatureManager.prepare_anomaly_data(df)

        return X_hedonic, X_anomaly


    ##########################################################
    # PRICE PREDICTION
    ##########################################################

    def predict_prices(
        self,
        df: pd.DataFrame,
    ):
        """
        Predict market price using Hedonic model.
        """

        logger.info("Predicting prices...")

        X_hedonic, _ = self.prepare_features(df)

        #######################################################
        # Predict log price
        #######################################################

        model_features = self.hedonic_model.feature_name_

        # Add missing columns with zero
        for col in model_features:
            if col not in X_hedonic.columns:
                X_hedonic[col] = 0

        # Remove extra columns and keep training order
        X_hedonic = X_hedonic[model_features]

        pred_log = self.hedonic_model.predict(
            X_hedonic
        )

        #######################################################
        # Convert back to price
        #######################################################

        predicted_price = np.expm1(

            pred_log

        )

        df["predicted_price"] = np.round(

            predicted_price,

            -2,

        )

        #######################################################
        # Residual
        #######################################################

        if "Price" in df.columns:

            df["residual_pct"] = (

                (

                    df["Price"]

                    -

                    df["predicted_price"]

                )

                /

                df["predicted_price"]

            )*100

        else:

            df["residual_pct"] = (

                (

                    df["predicted_price"]

                    -

                    df["loc_price_median"]

                )

                /

                df["loc_price_median"]

            )*100

        return df


    ##########################################################
    # ANOMALY PREDICTION
    ##########################################################

    def predict_anomalies(
        self,
        df: pd.DataFrame,
    ):
        """
        Predict anomaly score and label.
        """

        logger.info("Predicting anomalies...")


        #######################################################
        # CREATE FEATURES REQUIRED BY ISOLATION FOREST
        #######################################################

        if "price_per_sqft" not in df.columns:

            logger.info(
                "Creating price_per_sqft feature"
            )

            if "price_per_sqft" not in df.columns:

                if "Price" in df.columns:

                    df["price_per_sqft"] = (
                        df["Price"] /
                        df["Floor_Area"].replace(0,1)
                    )

                elif "predicted_price" in df.columns:

                    df["price_per_sqft"] = (
                        df["predicted_price"] /
                        df["Floor_Area"].replace(0,1)
                    )

                else:

                    raise ValueError(
                        "Cannot create price_per_sqft. "
                        "Price or predicted_price missing."
                    )


        #######################################################
        # PREPARE ANOMALY FEATURES
        #######################################################

        _, X_anomaly = self.prepare_features(df)

        # Ensure anomaly features match training features
        model_features = self.anomaly_features

        for col in model_features:
            if col not in X_anomaly.columns:
                X_anomaly[col] = 0

        X_anomaly = X_anomaly[model_features]

        #######################################################
        # Label
        #######################################################

        labels = self.anomaly_model.predict(

            X_anomaly

        )

        #######################################################
        # Score
        #######################################################

        scores = self.anomaly_model.score_samples(

            X_anomaly

        )

        df["anomaly_label"] = labels

        df["anomaly_score"] = scores
        df["anomaly_feat"] = df["anomaly_score"]

        # Residual feature for classifier/scoring
        if "residual_pct" in df.columns:
            
            df["residual_feat"] = df["residual_pct"]

        else:

            df["residual_feat"] = 0


        return df

    ##########################################################
    # CLASSIFIER FEATURES
    ##########################################################

    def prepare_classifier_features(
    self,
    df: pd.DataFrame,
    ):
        """
        Prepare classifier features.
        """

        logger.info(
            "Preparing classifier features..."
        )

        result = FeatureManager.prepare_classifier_data(
            df
        )

        # Handle tuple returns safely
        if isinstance(result, tuple):
            X_clf = result[0]
        else:
            X_clf = result

        return X_clf


    ##########################################################
    # TIER PREDICTION
    ##########################################################

    def predict_tiers(
        self,
        df: pd.DataFrame,
    ):
        """
        Predict investment tier using LightGBM classifier.
        """

        logger.info(

            "Predicting investment tier..."

        )

        X_clf = self.prepare_classifier_features(

            df

        )

        # Ensure classifier receives exactly the training features
        model_features = self.classifier.feature_name_

        # Add any missing columns
        for col in model_features:
            if col not in X_clf.columns:
                X_clf[col] = 0

        # Keep only training columns in the correct order
        X_clf = X_clf[model_features]
        
        
        
        predictions = self.classifier.predict(

            X_clf

        )

        tiers = self.encoder.inverse_transform(

            predictions

        )

        df["predicted_tier"] = tiers

        return df
    
    ##########################################################
    # SCORE ENTIRE DATAFRAME
    ##########################################################

    
    
    
    def score_dataframe(
        self,
        df: pd.DataFrame,
        include_classifier=True
    ) -> pd.DataFrame:
        """
        Complete inference pipeline.

        Keeps original business columns and appends ML predictions.
        """

        logger.info("=" * 70)
        logger.info("Running Complete Opportunity Scoring")
        logger.info("=" * 70)

        #######################################################
        # KEEP ORIGINAL DATA
        #######################################################

        original_df = df.copy(deep=True)

        #######################################################
        # CREATE MODEL FEATURES
        #######################################################

        # API sends raw columns:
        # Location, Property_Type, Condition, etc.
        # Training sends feature_engineered.csv:
        # Location_Boston, Property_Type_Apartment, etc.

        if "Location" in df.columns:

            logger.info(
                "Raw property input detected. Creating features..."
            )

            feature_df = FeatureManager.prepare_prediction_data(
                df.copy()
            )

        else:

            logger.info(
                "Feature engineered data detected. Using directly..."
            )

            feature_df = df.copy()


        #######################################################
        # LOAD MODELS
        #######################################################

        if self.hedonic_model is None:
            self.load_models(
                include_classifier=include_classifier
            )


        #######################################################
        # MODEL 1
        #######################################################

        feature_df = self.predict_prices(feature_df)


        #######################################################
        # MODEL 2
        #######################################################

        feature_df = self.predict_anomalies(feature_df)

        #######################################################
        # BUSINESS LOGIC
        #######################################################

        # Restore business columns required for scoring
        business_cols = [
            "Location",
            "Property_Type",
            "Crimerate",
            "accessibility",
            "loc_price_median",
            "prop_type_price_median"
        ]

        for col in business_cols:
            if col in original_df.columns:
                feature_df[col] = original_df[col]
        
        
        
        
        feature_df = self.calculate_factor_scores(feature_df)

        feature_df = self.calculate_opportunity_score(feature_df)

        feature_df = self.assign_tier(feature_df)

        feature_df = self.calculate_percentile(feature_df)


        #######################################################
        # MODEL 3
        #######################################################

        if include_classifier:

            feature_df = self.predict_tiers(feature_df)


        #######################################################
        # MERGE RESULTS BACK
        #######################################################

        prediction_columns = [

            "predicted_price",

            "residual_pct",

            "anomaly_score",

            "anomaly_label",

            "value_gap_score",

            "growth_score",

            "access_score",

            "safety_score",

            "anomaly_score_n",

            "opportunity_score",

            "tier",

            "percentile",

        ]

        for col in prediction_columns:

            if col in feature_df.columns:

                original_df[col] = feature_df[col]


        logger.info(
            "Scoring Completed Successfully"
        )


        return original_df
    
    
    
    
    
    
    
    
    
    
    
    
    

    ##########################################################
    # SCORE SINGLE PROPERTY
    ##########################################################

    def score_single_property(
        self,
        property_data: dict,
    ) -> dict:
        """
        Score one property.

        Useful for FastAPI prediction endpoint.
        """

        df = pd.DataFrame(

            [property_data]

        )

        scored = self.score_dataframe(

            df

        )

        return scored.iloc[0].to_dict()


    ##########################################################
    # SAVE SCORED DATASET
    ##########################################################

    def save_scored_dataset(
        self,
        df: pd.DataFrame,
        filename: str = "scored_properties.csv",
    ):
        """
        Save scored dataframe.
        """

        output_path = (

            ROOT_DIR

            / "data"

            / "processed"

            / filename

        )

        df.to_csv(

            output_path,

            index=False,

        )

        logger.info(

            f"Saved {output_path}"

        )


    ##########################################################
    # PREDICT (ALIAS)
    ##########################################################

    def predict(
        self,
        df: pd.DataFrame,
    ):
        """
        Alias for score_dataframe().
        """

        return self.score_dataframe(

            df

        )


    ##########################################################
    # STRING REPRESENTATION
    ##########################################################

    def __repr__(
        self,
    ):

        return (

            "OpportunityScorer("

            f"weights={len(self.weights)}, "

            f"models_loaded={self.hedonic_model is not None}"

            ")"

        )
        
        
    def run(self):

        data_path = ROOT_DIR / "data" / "processed" / "feature_engineered.csv"

        df = pd.read_csv(data_path)

        scored = self.score_dataframe(df,include_classifier=False)
                
        self.save_scored_dataset(scored)

        return scored


##############################################################
# MAIN
##############################################################

def main():

    data_path = (

        ROOT_DIR

        / "data"

        / "processed"

        / "feature_engineered.csv"

    )

    if not data_path.exists():

        raise FileNotFoundError(

            data_path

        )

    df = pd.read_csv(

        data_path

    )

    scorer = OpportunityScorer()

    scored_df = scorer.score_dataframe(

        df,
        include_classifier=False

    )

    scorer.save_scored_dataset(

        scored_df

    )

    print()

    print("=" * 70)

    print("Scoring Completed")

    print("=" * 70)

    print()
    columns = [
    "predicted_price",
    "residual_pct",
    "anomaly_score",
    "opportunity_score",
    "tier",
    ]

    if "predicted_tier" in scored_df.columns:
        columns.append("predicted_tier")

    print(scored_df[columns].head())

    # print(

    #     scored_df[

    #         [

    #             "predicted_price",

    #             "residual_pct",

    #             "anomaly_score",

    #             "opportunity_score",

    #             "tier",

    #             # "predicted_tier",

    #         ]

    #     ].head()

    # )


if __name__ == "__main__":

    main()