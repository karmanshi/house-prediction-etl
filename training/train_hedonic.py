"""
=============================================================
REIOS - Hedonic Price Model Trainer
=============================================================

Purpose
-------
Train the Hedonic Price Prediction model used by REIOS.

Pipeline
--------
1. Load engineered dataset
2. Prepare features
3. (Optional) Compare regression models
4. Train final LightGBM model
5. Evaluate model
6. Log experiment to MLflow
7. Save model
8. Save metrics
9. Save feature importance
10. Return predictions & residuals

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
import numpy as np
import pandas as pd
import yaml

from lightgbm import LGBMRegressor

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

class HedonicTrainer:
    """
    Production trainer for Hedonic Price Prediction.

    This class supports

    ✔ Data loading

    ✔ Model comparison

    ✔ Final LightGBM training

    ✔ Evaluation

    ✔ MLflow logging

    ✔ Model persistence

    ✔ Prediction
    """

    ##########################################################
    # INITIALIZATION
    ##########################################################

    def __init__(self):

        logger.info("=" * 60)
        logger.info("Initializing Hedonic Trainer")
        logger.info("=" * 60)

        ######################################################
        # CONFIGURATION
        ######################################################

        self.config = self._load_config()

        ######################################################
        # DATA
        ######################################################

        self.dataset: Optional[pd.DataFrame] = None

        self.feature_names = []

        self.X_train = None
        self.X_test = None

        self.y_train = None
        self.y_test = None

        ######################################################
        # MODEL
        ######################################################

        self.model: Optional[LGBMRegressor] = None

        ######################################################
        # RESULTS
        ######################################################

        self.metrics: Dict = {}

        self.results = None

        self.feature_importance = None

        self.predictions = None

        self.residuals = None

        ######################################################
        # MLFLOW
        ######################################################

        # mlflow.set_tracking_uri(

        #     f"file://{MLFLOW_DIR}"

        # )
        #mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

        mlflow.set_experiment(

            "REIOS_Hedonic_Regression"

        )

        logger.info("Initialization Complete")

    ##########################################################
    # CONFIGURATION
    ##########################################################

    def _load_config(self) -> Dict:
        """
        Load model configuration.
        """

        config_path = CONFIG_DIR / "model_config.yaml"

        if not config_path.exists():

            logger.warning(
                "model_config.yaml not found."
            )

            return {}

        with open(config_path, "r") as file:

            config = yaml.safe_load(file)

        logger.info("Configuration Loaded")

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

    from training.feature_manager import FeatureManager
    
    def prepare_data(
    self,
    test_size: float = 0.20,
    random_state: int = 42,
    ):
        """
        Prepare data using FeatureManager.
        """

        from sklearn.model_selection import train_test_split

        if self.dataset is None:

            raise RuntimeError(

                "Dataset not loaded."

            )

        logger.info("=" * 60)
        logger.info("Preparing Features")
        logger.info("=" * 60)

        #######################################################
        # FEATURE MANAGER
        #######################################################

        (

            X,

            y,

            self.feature_names,

        ) = FeatureManager.prepare_hedonic_data(

            self.dataset

        )

        #######################################################
        # TRAIN TEST SPLIT
        #######################################################

        (

            self.X_train,

            self.X_test,

            self.y_train,

            self.y_test,

        ) = train_test_split(

            X,

            y,

            test_size=test_size,

            random_state=random_state,

            shuffle=True,

        )

        logger.info(

            f"Training Samples : {len(self.X_train):,}"

        )

        logger.info(

            f"Testing Samples  : {len(self.X_test):,}"

        )

        logger.info(

            f"Feature Count    : {len(self.feature_names)}"

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

        print("HEDONIC DATA SUMMARY")

        print("=" * 60)

        print(

            f"Rows            : {len(self.dataset):,}"

        )

        print(

            f"Columns         : {len(self.dataset.columns)}"

        )

        print(

            f"Train Samples   : {len(self.X_train):,}"

        )

        print(

            f"Test Samples    : {len(self.X_test):,}"

        )

        print(

            f"Feature Count   : {len(self.feature_names)}"

        )

        print(

            f"Target          : {FeatureManager.TARGET}"

        )

        print("=" * 60)


    ##########################################################
    # MODEL PARAMETERS
    ##########################################################

    def get_model_params(
    self,
    ):
        """
        Read LightGBM parameters from YAML.
        """

        params = self.config.get(

            "hedonic_model",

            {}

        )

        defaults = {

            "n_estimators": 500,

            "learning_rate": 0.05,

            "max_depth": 6,

            "num_leaves": 63,

            "subsample": 0.80,

            "colsample_bytree": 0.80,

            "random_state": 42,

            "n_jobs": -1,

            "verbose": -1,

        }

        defaults.update(params)

        return defaults


    #for running models
    #trainer = HedonicTrainer()

    #trainer.load_dataset()

    #trainer.prepare_data()

    #trainer.dataset_summary()

    ##########################################################
    # CREATE LIGHTGBM MODEL
    ##########################################################

    def build_model(self):
        """
        Create the final LightGBM model from configuration.
        """

        logger.info("=" * 60)
        logger.info("Building LightGBM Model")
        logger.info("=" * 60)

        params = self.get_model_params()

        self.model = LGBMRegressor(**params)

        logger.info("Model Created Successfully")

        return self


    ##########################################################
    # TRAIN FINAL MODEL
    ##########################################################

    def train_model(self):
        """
        Train the production LightGBM model.
        """

        if self.model is None:

            self.build_model()

        logger.info("=" * 60)
        logger.info("Training Final Hedonic Model")
        logger.info("=" * 60)

        self.model.fit(

            self.X_train,

            self.y_train,

        )

        logger.info("Training Completed")

        return self


    ##########################################################
    # GENERATE PREDICTIONS
    ##########################################################

    def predict(self):
        """
        Predict log prices and actual prices.
        """

        if self.model is None:

            raise RuntimeError(

                "Model has not been trained."

            )

        #######################################################
        # LOG PREDICTIONS
        #######################################################

        pred_log = self.model.predict(

            self.X_test

        )

        #######################################################
        # ORIGINAL PRICE
        #######################################################

        pred_price = np.expm1(

            pred_log

        )

        actual_price = np.expm1(

            self.y_test

        )

        #######################################################
        # STORE
        #######################################################

        self.predictions = {

            "pred_log": pred_log,

            "pred_price": pred_price,

            "actual_price": actual_price,

        }

        logger.info(

            "Predictions Generated"

        )

        return self.predictions


    ##########################################################
    # CALCULATE RESIDUALS
    ##########################################################

    def calculate_residuals(self):
        """
        Calculate percentage residuals.

        Positive:
            Underpriced

        Negative:
            Overpriced
        """

        if self.predictions is None:

            self.predict()

        actual = self.predictions["actual_price"]

        predicted = self.predictions["pred_price"]

        residual_pct = (

            (

                actual -

                predicted

            )

            /

            predicted

        ) * 100

        self.residuals = residual_pct

        logger.info(

            "Residuals Calculated"

        )

        return residual_pct

    ##########################################################
    # EVALUATE MODEL
    ##########################################################

    def evaluate(self):
        """
        Evaluate trained model.
        """

        from sklearn.metrics import (
            r2_score,
            mean_absolute_error,
            mean_squared_error,
            mean_absolute_percentage_error,
        )

        logger.info("=" * 60)
        logger.info("Evaluating Model")
        logger.info("=" * 60)

        if self.predictions is None:

            self.predict()

        pred_log = self.predictions["pred_log"]

        pred_price = self.predictions["pred_price"]

        actual_price = self.predictions["actual_price"]

        #######################################################
        # METRICS
        #######################################################

        r2 = r2_score(

            self.y_test,

            pred_log,

        )

        mae = mean_absolute_error(

            actual_price,

            pred_price,

        )

        rmse = np.sqrt(

            mean_squared_error(

                actual_price,

                pred_price,

            )

        )

        mape = (

            mean_absolute_percentage_error(

                actual_price,

                pred_price,

            )

            * 100

        )

        self.metrics = {

            "r2": round(r2, 4),

            "mae": round(mae, 2),

            "rmse": round(rmse, 2),

            "mape": round(mape, 2),

        }

        logger.info(self.metrics)

        return self.metrics


    ##########################################################
    # SAVE METRICS
    ##########################################################

    def save_metrics(self):
        """
        Save evaluation metrics.
        """

        metric_path = (

            METRIC_DIR /

            "hedonic_metrics.json"

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
    # FEATURE IMPORTANCE
    ##########################################################

    def save_feature_importance(self):
        """
        Save feature importance.
        """

        import matplotlib.pyplot as plt

        importance = pd.DataFrame({

            "Feature": self.feature_names,

            "Importance": self.model.feature_importances_,

        })

        importance = importance.sort_values(

            "Importance",

            ascending=False,

        )

        self.feature_importance = importance

        #######################################################
        # CSV
        #######################################################

        csv_path = (

            PLOT_DIR /

            "feature_importance.csv"

        )

        importance.to_csv(

            csv_path,

            index=False,

        )

        #######################################################
        # PLOT
        #######################################################

        plt.figure(

            figsize=(10,8)

        )

        plt.barh(

            importance["Feature"][:20],

            importance["Importance"][:20],

        )

        plt.gca().invert_yaxis()

        plt.title(

            "Top 20 Feature Importance"

        )

        plt.tight_layout()

        plot_path = (

            PLOT_DIR /

            "feature_importance.png"

        )

        plt.savefig(

            plot_path,

            dpi=200,

        )

        plt.close()

        logger.info(

            "Feature Importance Saved"

        )


    ##########################################################
    # RESIDUAL ANALYSIS
    ##########################################################

    def save_residual_plot(self):
        """
        Save residual analysis plot.
        """

        import matplotlib.pyplot as plt

        plt.figure(

            figsize=(8,6)

        )

        plt.scatter(

            self.predictions["pred_price"],

            self.residuals,

            alpha=0.30,

        )

        plt.axhline(

            y=0,

            color="red",

            linestyle="--",

        )

        plt.xlabel(

            "Predicted Price"

        )

        plt.ylabel(

            "Residual (%)"

        )

        plt.title(

            "Residual Analysis"

        )

        plt.tight_layout()

        plt.savefig(

            PLOT_DIR /

            "residual_analysis.png",

            dpi=200,

        )

        plt.close()

        logger.info(

            "Residual Plot Saved"

        )


    ##########################################################
    # SAVE MODEL
    ##########################################################

    def save_model(self):
        """
        Save trained model.
        """

        model_path = (

            MODEL_DIR /

            "lgbm_hedonic.pkl"

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

    def log_mlflow(self):
        """
        Log experiment.
        """

        with mlflow.start_run(

            run_name="hedonic_final"

        ):

            mlflow.log_params(

                self.get_model_params()

            )

            mlflow.log_metrics(

                self.metrics

            )

            mlflow.lightgbm.log_model(

                self.model,

                artifact_path="lgbm_hedonic",

            )

            mlflow.log_artifact(

                str(

                    METRIC_DIR /

                    "hedonic_metrics.json"

                )

            )

            mlflow.log_artifact(

                str(

                    PLOT_DIR /

                    "feature_importance.csv"

                )

            )

            mlflow.log_artifact(

                str(

                    PLOT_DIR /

                    "feature_importance.png"

                )

            )

            mlflow.log_artifact(

                str(

                    PLOT_DIR /

                    "residual_analysis.png"

                )

            )

            logger.info(

                "MLflow Logging Complete"

            )
            
        
    ##########################################################
    # LOAD TRAINED MODEL
    ##########################################################

    def load_model(self):
        """
        Load trained LightGBM model.
        """

        model_path = MODEL_DIR / "lgbm_hedonic.pkl"

        if not model_path.exists():

            raise FileNotFoundError(

                f"Model not found:\n{model_path}"

            )

        self.model = joblib.load(model_path)

        logger.info("LightGBM model loaded successfully.")

        return self


    ##########################################################
    # PREDICT NEW DATA
    ##########################################################

    def predict_new(
    self,
    X: pd.DataFrame,
    ):
        """
        Predict property prices for new data.

        Parameters
        ----------
        X : DataFrame
            Feature matrix prepared using
            FeatureManager.get_hedonic_features()

        Returns
        -------
        DataFrame
        """

        if self.model is None:

            self.load_model()

        X = FeatureManager.clean_features(X)

        pred_log = self.model.predict(X)

        pred_price = np.expm1(pred_log)

        return pd.DataFrame({

            "predicted_price": pred_price,

        })


    ##########################################################
    # FULL TRAINING PIPELINE
    ##########################################################

    def train(
    self,
    log_mlflow: bool = True,
    ):
        
        """
        Execute complete production training pipeline.
        """

        logger.info("=" * 70)
        logger.info("REIOS Hedonic Training Pipeline")
        logger.info("=" * 70)

        (

            self.load_dataset()

                .prepare_data()

                .build_model()

                .train_model()

        )

        self.predict()

        self.calculate_residuals()

        self.evaluate()

        self.save_metrics()

        self.save_feature_importance()

        self.save_residual_plot()

        self.save_model()

        self.log_mlflow()

        logger.info("=" * 70)
        logger.info("Training Pipeline Completed Successfully")
        logger.info("=" * 70)

        return {

            "model": self.model,

            "metrics": self.metrics,

            "predictions": self.predictions,

            "residuals": self.residuals,

        }


    ##########################################################
    # STRING REPRESENTATION
    ##########################################################

    def __repr__(self):

        return (

            f"HedonicTrainer("

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

    trainer = HedonicTrainer()

    trainer.train()


if __name__ == "__main__":

    main()