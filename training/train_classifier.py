"""
=============================================================
REIOS - Investment Tier Classifier Trainer
=============================================================

Purpose
-------
Train the LightGBM classifier used to predict the
investment tier of a property.

Pipeline
--------
1. Load scored dataset
2. Prepare classifier features
3. Train LightGBM Classifier
4. Evaluate classifier
5. Log experiment to MLflow
6. Save trained model
7. Save Label Encoder
8. Save evaluation metrics

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
import mlflow.lightgbm
import pandas as pd
import yaml

from lightgbm import LGBMClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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

class ClassifierTrainer:
    """
    Production trainer for investment tier prediction.

    Responsibilities
    ----------------
    ✔ Load scored dataset

    ✔ Prepare classifier features

    ✔ Encode target labels

    ✔ Train LightGBM classifier

    ✔ Evaluate model

    ✔ Save model

    ✔ Save LabelEncoder

    ✔ MLflow logging
    """

    ##########################################################
    # INITIALIZATION
    ##########################################################

    def __init__(self):

        logger.info("=" * 60)
        logger.info("Initializing Classifier Trainer")
        logger.info("=" * 60)

        ######################################################
        # CONFIG
        ######################################################

        self.config = self._load_config()

        ######################################################
        # DATA
        ######################################################

        self.dataset: Optional[pd.DataFrame] = None

        self.X_train = None
        self.X_test = None

        self.y_train = None
        self.y_test = None

        self.feature_names = []

        ######################################################
        # MODEL
        ######################################################

        self.model: Optional[LGBMClassifier] = None

        self.label_encoder = LabelEncoder()

        ######################################################
        # METRICS
        ######################################################

        self.metrics: Dict = {}

        ######################################################
        # MLFLOW
        ######################################################

        # mlflow.set_tracking_uri(

        #     f"file://{MLFLOW_DIR}"

        # )
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(

            "REIOS_Tier_Classifier"

        )

        logger.info("Initialization Complete")

    ##########################################################
    # LOAD CONFIG
    ##########################################################

    def _load_config(self):

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
        Load scored dataset generated by scorer.py.
        """

        dataset_path = (

            DATA_DIR /

            "scored_properties.csv"

        )

        if not dataset_path.exists():

            raise FileNotFoundError(

                f"\nDataset not found\n{dataset_path}"

            )

        logger.info("=" * 60)
        logger.info("Loading Scored Dataset")
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
        Prepare classifier features and labels.
        """

        if self.dataset is None:

            raise RuntimeError(

                "Dataset not loaded."

            )

        logger.info("=" * 60)
        logger.info("Preparing Classifier Dataset")
        logger.info("=" * 60)

        ######################################################
        # FEATURES
        ######################################################

        X, _,self.feature_names = (

            FeatureManager.prepare_classifier_data(

                self.dataset

            )

        )

        ######################################################
        # TARGET
        ######################################################

        if "tier" not in self.dataset.columns:

            raise ValueError(

                "'tier' column missing."

            )

        y = self.label_encoder.fit_transform(

            self.dataset["tier"]

        )

        ######################################################
        # TRAIN TEST SPLIT
        ######################################################

        (

            self.X_train,

            self.X_test,

            self.y_train,

            self.y_test,

        ) = train_test_split(

            X,

            y,

            test_size=0.20,

            random_state=42,

            stratify=y,

        )

        logger.info(

            "Classifier Dataset Ready"

        )

        return self


    ##########################################################
    # DATA SUMMARY
    ##########################################################

    def dataset_summary(
        self,
    ):
        """
        Display dataset summary.
        """

        if self.dataset is None:

            raise RuntimeError(

                "Dataset not loaded."

            )

        print()

        print("=" * 60)

        print("CLASSIFIER DATA SUMMARY")

        print("=" * 60)

        print(

            f"Rows            : {len(self.dataset):,}"

        )

        print(

            f"Columns         : {len(self.dataset.columns)}"

        )

        print(

            f"Training Samples: {len(self.X_train):,}"

        )

        print(

            f"Testing Samples : {len(self.X_test):,}"

        )

        print(

            f"Features        : {len(self.feature_names)}"

        )

        print(

            "Classes         :",

            list(self.label_encoder.classes_)

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
        Read LightGBM classifier parameters
        from model_config.yaml.
        """

        params = self.config.get(

            "classifier",

            {}

        )

        defaults = {

            "n_estimators": 300,

            "learning_rate": 0.05,

            "max_depth": 5,

            "num_leaves": 31,

            "random_state": 42,

            "n_jobs": -1,

            "verbose": -1,

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
        Create LightGBM Classifier.
        """

        logger.info("=" * 60)
        logger.info("Building LightGBM Classifier")
        logger.info("=" * 60)

        params = self.get_model_params()

        self.model = LGBMClassifier(

            **params

        )

        logger.info(

            "LightGBM Classifier Created"

        )

        return self


    ##########################################################
    # TRAIN MODEL
    ##########################################################

    def train_model(
        self,
    ):
        """
        Train LightGBM classifier.
        """

        if self.model is None:

            self.build_model()

        logger.info("=" * 60)
        logger.info("Training Classifier")
        logger.info("=" * 60)

        self.model.fit(

            self.X_train,

            self.y_train,

        )

        logger.info(

            "Training Completed"

        )

        return self


    ##########################################################
    # EVALUATE MODEL
    ##########################################################

    def evaluate_model(
        self,
    ):
        """
        Evaluate classifier performance.
        """

        from sklearn.metrics import (

            accuracy_score,

            f1_score,

            classification_report,

            confusion_matrix,

        )

        import matplotlib.pyplot as plt
        import seaborn as sns

        logger.info("=" * 60)
        logger.info("Evaluating Classifier")
        logger.info("=" * 60)

        #######################################################
        # PREDICTIONS
        #######################################################

        predictions = self.model.predict(

            self.X_test

        )

        #######################################################
        # METRICS
        #######################################################

        accuracy = accuracy_score(

            self.y_test,

            predictions,

        )

        weighted_f1 = f1_score(

            self.y_test,

            predictions,

            average="weighted",

        )

        report = classification_report(

            self.y_test,

            predictions,

            target_names=self.label_encoder.classes_,

            output_dict=True,

        )

        #######################################################
        # STORE METRICS
        #######################################################

        self.metrics = {

            "accuracy": round(

                accuracy,

                4,

            ),

            "weighted_f1": round(

                weighted_f1,

                4,

            ),

        }

        #######################################################
        # PER CLASS F1
        #######################################################

        for cls in self.label_encoder.classes_:

            self.metrics[

                f"f1_{cls}"

            ] = round(

                report[cls]["f1-score"],

                4,

            )

        #######################################################
        # CONFUSION MATRIX
        #######################################################

        cm = confusion_matrix(

            self.y_test,

            predictions,

        )

        plt.figure(

            figsize=(7,6)

        )

        sns.heatmap(

            cm,

            annot=True,

            fmt="d",

            cmap="Blues",

            xticklabels=self.label_encoder.classes_,

            yticklabels=self.label_encoder.classes_,

        )

        plt.xlabel(

            "Predicted"

        )

        plt.ylabel(

            "Actual"

        )

        plt.title(

            "Investment Tier Confusion Matrix"

        )

        plt.tight_layout()

        plt.savefig(

            PLOT_DIR /

            "confusion_matrix.png",

            dpi=200,

        )

        plt.close()

        logger.info(

            "Evaluation Completed"

        )

        return self.metrics  
    
    #########################################################
    # SAVE MODEL
    ##########################################################

    def save_model(
        self,
    ):
        """
        Save trained LightGBM classifier.
        """

        model_path = (

            MODEL_DIR /

            "lgbm_classifier.pkl"

        )

        joblib.dump(

            self.model,

            model_path,

        )

        logger.info(

            f"Saved {model_path}"

        )


    ##########################################################
    # SAVE LABEL ENCODER
    ##########################################################

    def save_encoder(
        self,
    ):
        """
        Save LabelEncoder.
        """

        encoder_path = (

            MODEL_DIR /

            "tier_encoder.pkl"

        )

        joblib.dump(

            self.label_encoder,

            encoder_path,

        )

        logger.info(

            f"Saved {encoder_path}"

        )


    ##########################################################
    # SAVE METRICS
    ##########################################################

    def save_metrics(
        self,
    ):
        """
        Save evaluation metrics.
        """

        metric_path = (

            METRIC_DIR /

            "classifier_metrics.json"

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
    # LOG MLFLOW
    ##########################################################

    def log_mlflow(
        self,
    ):
        """
        Log classifier experiment.
        """

        with mlflow.start_run(

            run_name="lgbm_classifier_final"

        ):

            ####################################################
            # PARAMETERS
            ####################################################

            mlflow.log_params(

                self.get_model_params()

            )

            ####################################################
            # METRICS
            ####################################################

            mlflow.log_metrics(

                self.metrics

            )

            ####################################################
            # MODEL
            ####################################################

            mlflow.lightgbm.log_model(

                self.model,

                artifact_path="lgbm_classifier",

            )

            ####################################################
            # ARTIFACTS
            ####################################################

            metric_file = (

                METRIC_DIR /

                "classifier_metrics.json"

            )

            cm_plot = (

                PLOT_DIR /

                "confusion_matrix.png"

            )

            if metric_file.exists():

                mlflow.log_artifact(

                    str(metric_file)

                )

            if cm_plot.exists():

                mlflow.log_artifact(

                    str(cm_plot)

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
        Load trained classifier and encoder.
        """

        self.model = joblib.load(

            MODEL_DIR /

            "lgbm_classifier.pkl"

        )

        self.label_encoder = joblib.load(

            MODEL_DIR /

            "tier_encoder.pkl"

        )

        logger.info(

            "Classifier Loaded"

        )

        return self


    ##########################################################
    # PREDICT NEW DATA
    ##########################################################

    def predict_new(
        self,
        df: pd.DataFrame,
    ):
        """
        Predict investment tier.

        Parameters
        ----------
        df : DataFrame already containing
            classifier features.

        Returns
        -------
        DataFrame
        """

        if self.model is None:

            self.load_model()

        X, _ = FeatureManager.prepare_classifier_data(

            df

        )

        predictions = self.model.predict(

            X

        )

        tiers = self.label_encoder.inverse_transform(

            predictions

        )

        result = df.copy()

        result["predicted_tier"] = tiers

        return result


    ##########################################################
    # COMPLETE TRAINING PIPELINE
    ##########################################################

    def train(
        self,
    ):
        """
        Execute complete classifier pipeline.
        """

        logger.info("=" * 70)
        logger.info("REIOS Classifier Training Pipeline")
        logger.info("=" * 70)

        (

            self.load_dataset()

                .prepare_data()

                .build_model()

                .train_model()

        )

        self.evaluate_model()

        self.save_model()

        self.save_encoder()

        self.save_metrics()

        self.log_mlflow()

        logger.info("=" * 70)
        logger.info("Training Completed Successfully")
        logger.info("=" * 70)

        return {

            "model": self.model,

            "metrics": self.metrics,

        }


    ##########################################################
    # STRING REPRESENTATION
    ##########################################################

    def __repr__(
        self,
    ):

        return (

            "ClassifierTrainer("

            f"features={len(self.feature_names)}, "

            f"trained={self.model is not None}"

            ")"

        )
    
##############################################################
# MAIN
##############################################################

def main():

    trainer = ClassifierTrainer()

    trainer.train()


if __name__ == "__main__":

    main()      