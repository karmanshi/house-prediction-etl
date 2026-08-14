"""
=============================================================
REIOS - Anomaly Detection Trainer
=============================================================

Purpose
-------
Train the Isolation Forest model used to detect unusual
real estate listings.

Pipeline
--------
1. Load engineered dataset
2. Prepare anomaly features
3. Train Isolation Forest
4. Generate anomaly scores
5. Generate anomaly labels
6. Log experiment to MLflow
7. Save trained model
8. Save anomaly metrics

Author
------
Bhavya

=============================================================
"""

from __future__ import annotations

##############################################################
# STANDARD LIBRARIES
##############################################################

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, Optional

##############################################################
# THIRD PARTY LIBRARIES
##############################################################

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml

from sklearn.ensemble import IsolationForest

##############################################################
# PROJECT MODULES
##############################################################

from training.feature_manager import FeatureManager

warnings.filterwarnings("ignore")

##############################################################
# PROJECT PATHS
##############################################################

ROOT_DIR = Path(__file__).resolve().parent.parent

CONFIG_DIR = ROOT_DIR / "configs"

DATA_DIR = ROOT_DIR / "data" / "processed"

MODEL_DIR = ROOT_DIR / "models"

METRIC_DIR = ROOT_DIR / "metrics"

PLOT_DIR = ROOT_DIR / "evaluation_plots"

MLFLOW_DIR = ROOT_DIR / "mlruns"

##############################################################
# CREATE DIRECTORIES
##############################################################

MODEL_DIR.mkdir(parents=True, exist_ok=True)

METRIC_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

##############################################################
# LOGGING
##############################################################

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s",

)

logger = logging.getLogger(__name__)

##############################################################
# TRAINER
##############################################################

class AnomalyTrainer:
    """
    Production trainer for Isolation Forest.

    Responsibilities

    ✔ Load dataset

    ✔ Prepare anomaly features

    ✔ Train Isolation Forest

    ✔ Generate anomaly scores

    ✔ Save model

    ✔ Log MLflow

    ✔ Inference
    """

    ##########################################################
    # INITIALIZATION
    ##########################################################

    def __init__(self):

        logger.info("=" * 60)
        logger.info("Initializing Anomaly Trainer")
        logger.info("=" * 60)

        ######################################################
        # CONFIG
        ######################################################

        self.config = self._load_config()

        ######################################################
        # DATA
        ######################################################

        self.dataset: Optional[pd.DataFrame] = None

        self.feature_names = []

        self.X = None

        ######################################################
        # MODEL
        ######################################################

        self.model: Optional[IsolationForest] = None

        ######################################################
        # OUTPUTS
        ######################################################

        self.anomaly_scores = None

        self.anomaly_labels = None

        self.metrics: Dict = {}

        ######################################################
        # MLFLOW
        ######################################################

        # mlflow.set_tracking_uri(

        #     f"file://{MLFLOW_DIR}"

        # )
        
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

        mlflow.set_experiment(

            "REIOS_Anomaly_Detection"

        )

        logger.info("Initialization Complete")

    ##########################################################
    # CONFIG
    ##########################################################

    def _load_config(
        self,
    ) -> Dict:
        """
        Load configuration.
        """

        config_path = (

            CONFIG_DIR /

            "model_config.yaml"

        )

        if not config_path.exists():

            logger.warning(

                "model_config.yaml not found."

            )

            return {}

        with open(

            config_path,

            "r",

        ) as fp:

            config = yaml.safe_load(fp)

        logger.info(

            "Configuration Loaded"

        )

        return config
    
    ##########################################################
    # LOAD DATASET
    ##########################################################

    def load_dataset(
        self,
    ):
        """
        Load engineered dataset.
        """

        dataset_path = (

            DATA_DIR /

            "feature_engineered.csv"

        )

        if not dataset_path.exists():

            raise FileNotFoundError(

                f"\nDataset not found\n{dataset_path}"

            )

        logger.info("=" * 60)
        logger.info("Loading Dataset")
        logger.info("=" * 60)

        self.dataset = pd.read_csv(

            dataset_path

        )

        logger.info(

            f"Dataset Shape : {self.dataset.shape}"

        )

        return self


    ##########################################################
    # PREPARE DATA
    ##########################################################

    def prepare_data(
        self,
    ):
        """
        Prepare features for anomaly detection.
        """

        if self.dataset is None:

            raise RuntimeError(

                "Dataset not loaded."

            )

        logger.info("=" * 60)
        logger.info("Preparing Anomaly Features")
        logger.info("=" * 60)

        (

            self.X,

            self.feature_names,

        ) = FeatureManager.prepare_anomaly_data(

            self.dataset

        )

        logger.info(

            f"Samples       : {len(self.X):,}"

        )

        logger.info(

            f"Feature Count : {len(self.feature_names)}"

        )

        return self


    ##########################################################
    # DATA SUMMARY
    ##########################################################

    def dataset_summary(
        self,
    ):
        """
        Print dataset summary.
        """

        if self.dataset is None:

            raise RuntimeError(

                "Dataset not loaded."

            )

        print()

        print("=" * 60)

        print("ANOMALY DATA SUMMARY")

        print("=" * 60)

        print(

            f"Rows            : {len(self.dataset):,}"

        )

        print(

            f"Columns         : {len(self.dataset.columns)}"

        )

        print(

            f"Samples         : {len(self.X):,}"

        )

        print(

            f"Feature Count   : {len(self.feature_names)}"

        )

        print("=" * 60)

        return self


    ##########################################################
    # MODEL PARAMETERS
    ##########################################################

    def get_model_params(
        self,
    ):
        """
        Read Isolation Forest parameters from YAML.
        """

        params = self.config.get(

            "anomaly_model",

            {}

        )

        defaults = {

            "n_estimators": 100,

            "contamination": 0.05,

            "random_state": 42,

            "n_jobs": -1,

        }

        defaults.update(

            params

        )

        return defaults
        
        
    ##########################################################
    # BUILD MODEL
    ##########################################################

    def build_model(
        self,
    ):
        """
        Create Isolation Forest model.
        """

        logger.info("=" * 60)
        logger.info("Building Isolation Forest")
        logger.info("=" * 60)

        params = self.get_model_params()

        self.model = IsolationForest(

            **params

        )

        logger.info(

            "Isolation Forest Created"

        )

        return self


    ##########################################################
    # TRAIN MODEL
    ##########################################################

    def train_model(
        self,
    ):
        """
        Train Isolation Forest.
        """

        if self.model is None:

            self.build_model()

        logger.info("=" * 60)
        logger.info("Training Isolation Forest")
        logger.info("=" * 60)

        self.model.fit(

            self.X

        )

        import json


        with open(
            "models/anomaly_features.json",
            "w"
        ) as f:

            json.dump(
                list(self.X.columns),
                f
            )
        
        
        
        
        
        logger.info(

            "Training Completed"

        )

        return self


    ##########################################################
    # DETECT ANOMALIES
    ##########################################################

    def detect_anomalies(
        self,
    ):
        """
        Generate anomaly labels and anomaly scores.

        Labels

            1  -> Normal

        -1  -> Anomaly
        """

        if self.model is None:

            raise RuntimeError(

                "Model has not been trained."

            )

        logger.info("=" * 60)
        logger.info("Detecting Anomalies")
        logger.info("=" * 60)

        #######################################################
        # LABELS
        #######################################################

        self.anomaly_labels = self.model.predict(

            self.X

        )

        #######################################################
        # SCORES
        #######################################################

        self.anomaly_scores = self.model.score_samples(

            self.X

        )

        logger.info(

            "Anomaly Detection Completed"

        )

        return (

            self.anomaly_labels,

            self.anomaly_scores,

        )


    ##########################################################
    # CALCULATE STATISTICS
    ##########################################################

    def calculate_statistics(
        self,
    ):
        """
        Compute anomaly statistics.
        """

        if self.anomaly_labels is None:

            self.detect_anomalies()

        #######################################################
        # COUNTS
        #######################################################

        anomaly_count = int(

            np.sum(

                self.anomaly_labels == -1

            )

        )

        normal_count = int(

            np.sum(

                self.anomaly_labels == 1

            )

        )

        anomaly_rate = (

            anomaly_count

            /

            len(self.anomaly_labels)

        ) * 100

        #######################################################
        # PRICE COMPARISON
        #######################################################

        if "price_per_sqft" in self.dataset.columns:

            temp = self.dataset.copy()

            temp["anomaly_label"] = self.anomaly_labels

            normal_price = (

                temp

                .loc[
                    temp["anomaly_label"] == 1,
                    "price_per_sqft"
                ]

                .mean()

            )

            anomaly_price = (

                temp

                .loc[
                    temp["anomaly_label"] == -1,
                    "price_per_sqft"
                ]

                .mean()

            )

        else:

            normal_price = None

            anomaly_price = None

        #######################################################
        # METRICS
        #######################################################

        self.metrics = {

            "total_samples": len(self.dataset),

            "normal_properties": normal_count,

            "anomalies_found": anomaly_count,

            "anomaly_rate": round(

                anomaly_rate,

                2,

            ),

            "avg_normal_price_per_sqft":

                None if normal_price is None

                else round(normal_price, 2),

            "avg_anomaly_price_per_sqft":

                None if anomaly_price is None

                else round(anomaly_price, 2),

        }

        logger.info(

            f"Anomalies Found : {anomaly_count:,}"

        )

        logger.info(

            f"Anomaly Rate    : {anomaly_rate:.2f}%"

        )

        return self.metrics
        ##########################################################
    # SAVE METRICS
    ##########################################################

    def save_metrics(
        self,
    ):
        """
        Save anomaly metrics.
        """

        metric_path = (

            METRIC_DIR /

            "anomaly_metrics.json"

        )

        with open(

            metric_path,

            "w",

        ) as fp:

            json.dump(

                self.metrics,

                fp,

                indent=4,

            )

        logger.info(

            f"Saved {metric_path}"

        )


    ##########################################################
    # SAVE SCORE DISTRIBUTION
    ##########################################################

    def save_distribution_plot(
        self,
    ):
        """
        Save anomaly score distribution.
        """

        import matplotlib.pyplot as plt

        plt.figure(

            figsize=(8,5)

        )

        plt.hist(

            self.anomaly_scores,

            bins=60,

        )

        plt.xlabel(

            "Isolation Forest Score"

        )

        plt.ylabel(

            "Count"

        )

        plt.title(

            "Anomaly Score Distribution"

        )

        plt.tight_layout()

        plt.savefig(

            PLOT_DIR /

            "anomaly_score_distribution.png",

            dpi=200,

        )

        plt.close()

        logger.info(

            "Distribution Plot Saved"

        )


    ##########################################################
    # SAVE MODEL
    ##########################################################

    def save_model(
        self,
    ):
        """
        Save trained Isolation Forest.
        """

        model_path = (

            MODEL_DIR /

            "iso_forest.pkl"

        )

        joblib.dump(

            self.model,

            model_path,

        )

        logger.info(

            f"Saved {model_path}"

        )


    ##########################################################
    # LOG TO MLFLOW
    ##########################################################

    def log_mlflow(
        self,
    ):
        """
        Log experiment.
        """

        with mlflow.start_run(

            run_name="isolation_forest_final"

        ):

            mlflow.log_params(

                self.get_model_params()

            )

            mlflow.log_metrics(

                self.metrics

            )

            mlflow.sklearn.log_model(

                self.model,

                artifact_path="iso_forest",

            )

            mlflow.log_artifact(

                str(

                    METRIC_DIR /

                    "anomaly_metrics.json"

                )

            )

            mlflow.log_artifact(

                str(

                    PLOT_DIR /

                    "anomaly_score_distribution.png"

                )

            )

            logger.info(

                "MLflow Logging Completed"

            )


    ##########################################################
    # LOAD TRAINED MODEL
    ##########################################################

    def load_model(
        self,
    ):
        """
        Load trained Isolation Forest.
        """

        model_path = (

            MODEL_DIR /

            "iso_forest.pkl"

        )

        if not model_path.exists():

            raise FileNotFoundError(

                f"Model not found\n{model_path}"

            )

        self.model = joblib.load(

            model_path

        )

        logger.info(

            "Isolation Forest Loaded"

        )

        return self


    ##########################################################
    # PREDICT NEW DATA
    ##########################################################

    def predict_new(
        self,
        X: pd.DataFrame,
    ):
        """
        Predict anomaly labels and scores
        for new properties.

        Returns
        -------
        DataFrame
        """

        if self.model is None:

            self.load_model()

        X = FeatureManager.clean_features(X)

        labels = self.model.predict(

            X

        )

        scores = self.model.score_samples(

            X

        )

        return pd.DataFrame({

            "anomaly_label": labels,

            "anomaly_score": scores,

        })


    ##########################################################
    # COMPLETE TRAINING PIPELINE
    ##########################################################

    def train(
        self,
    ):
        """
        Execute production anomaly pipeline.
        """

        logger.info("=" * 70)
        logger.info("REIOS Anomaly Training Pipeline")
        logger.info("=" * 70)

        (

            self.load_dataset()

                .prepare_data()

                .build_model()

                .train_model()

        )

        self.detect_anomalies()

        self.calculate_statistics()

        self.save_metrics()

        self.save_distribution_plot()

        self.save_model()

        self.log_mlflow()

        logger.info("=" * 70)
        logger.info("Training Completed Successfully")
        logger.info("=" * 70)

        return {

            "model": self.model,

            "labels": self.anomaly_labels,

            "scores": self.anomaly_scores,

            "metrics": self.metrics,

        }


    ##########################################################
    # STRING REPRESENTATION
    ##########################################################

    def __repr__(
        self,
    ):

        return (

            f"AnomalyTrainer("

            f"features={len(self.feature_names)}, "

            f"trained={self.model is not None}"

            f")"

        )
        
        
##############################################################
# MAIN
##############################################################

def main():
    """
    CLI Entry Point.
    """

    trainer = AnomalyTrainer()

    trainer.train()


if __name__ == "__main__":

    main()
    
    
    