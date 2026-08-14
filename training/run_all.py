"""
=============================================================
REIOS - Complete Training Pipeline
=============================================================

Purpose
-------
Runs the complete production ML pipeline.

Pipeline
--------
1. Hedonic Price Model
2. Anomaly Detection Model
3. Opportunity Scoring
4. Tier Classification Model

Outputs
-------
models/
    lgbm_hedonic.pkl
    iso_forest.pkl
    lgbm_classifier.pkl
    tier_encoder.pkl

metrics/

evaluation_plots/

mlruns/

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
import time
from pathlib import Path

##############################################################
# PROJECT MODULES
##############################################################

from training.train_hedonic import HedonicTrainer
from training.train_anomaly import AnomalyTrainer
from training.scorer import OpportunityScorer
from training.train_classifier import ClassifierTrainer

##############################################################
# PATHS
##############################################################

ROOT_DIR = Path(__file__).resolve().parent.parent

##############################################################
# LOGGING
##############################################################

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s",

)

logger = logging.getLogger(__name__)


##############################################################
# PIPELINE
##############################################################

class REIOSTrainingPipeline:
    """
    Complete production training pipeline.

    Order
    -----

    Hedonic Model
            ↓
    Isolation Forest
            ↓
    Composite Scorer
            ↓
    Tier Classifier
    """

    ##########################################################
    # INITIALIZATION
    ##########################################################

    def __init__(self):

        logger.info("=" * 70)
        logger.info("Initializing REIOS Training Pipeline")
        logger.info("=" * 70)

        self.start_time = None

        self.results = {}

        logger.info("Pipeline Ready")
        
        
    ##########################################################
    # RUN HEDONIC MODEL
    ##########################################################

    def run_hedonic(
        self,
    ):
        """
        Train Hedonic Price Prediction Model.
        """

        logger.info("=" * 70)
        logger.info("STEP 1 : Hedonic Price Model")
        logger.info("=" * 70)

        start = time.time()

        trainer = HedonicTrainer()

        result = trainer.train()

        elapsed = round(

            time.time() - start,

            2,

        )

        self.results["hedonic"] = {

            "status": "SUCCESS",

            "time": elapsed,

            "metrics": result.get(

                "metrics",

                {},

            ),

        }

        logger.info(

            f"Hedonic Completed ({elapsed}s)"

        )


    ##########################################################
    # RUN ANOMALY MODEL
    ##########################################################

    def run_anomaly(
        self,
    ):
        """
        Train Isolation Forest.
        """

        logger.info("=" * 70)
        logger.info("STEP 2 : Isolation Forest")
        logger.info("=" * 70)

        start = time.time()

        trainer = AnomalyTrainer()

        result = trainer.train()

        elapsed = round(

            time.time() - start,

            2,

        )

        self.results["anomaly"] = {

            "status": "SUCCESS",

            "time": elapsed,

            "metrics": result.get(

                "metrics",

                {},

            ),

        }

        logger.info(

            f"Anomaly Completed ({elapsed}s)"

        )


    ##########################################################
    # RUN COMPOSITE SCORER
    ##########################################################

    def run_scoring(
        self,
    ):
        """
        Generate scored_properties.csv
        """

        logger.info("=" * 70)
        logger.info("STEP 3 : Composite Scoring")
        logger.info("=" * 70)

        start = time.time()

        scorer = OpportunityScorer()

        result = scorer.run()

        elapsed = round(

            time.time() - start,

            2,

        )

        self.results["scoring"] = {
            "status": "SUCCESS",
            "time": elapsed,
            "rows": len(result),
        }

        logger.info(

            f"Scoring Completed ({elapsed}s)"

        )


    ##########################################################
    # RUN CLASSIFIER
    ##########################################################

    def run_classifier(
        self,
    ):
        """
        Train Investment Tier Classifier.
        """

        logger.info("=" * 70)
        logger.info("STEP 4 : Tier Classifier")
        logger.info("=" * 70)

        start = time.time()

        trainer = ClassifierTrainer()

        result = trainer.train()

        elapsed = round(

            time.time() - start,

            2,

        )

        self.results["classifier"] = {

            "status": "SUCCESS",

            "time": elapsed,

            "metrics": result.get(

                "metrics",

                {},

            ),

        }

        logger.info(

            f"Classifier Completed ({elapsed}s)"

        )
        
    ##########################################################
    # RUN COMPLETE PIPELINE
    ##########################################################

    def run(
        self,
    ):
        """
        Execute the complete REIOS training pipeline.
        """

        logger.info("=" * 70)
        logger.info("STARTING COMPLETE REIOS PIPELINE")
        logger.info("=" * 70)

        self.start_time = time.time()

        try:

            ##################################################
            # STEP 1
            ##################################################

            self.run_hedonic()

            ##################################################
            # STEP 2
            ##################################################

            self.run_anomaly()

            ##################################################
            # STEP 3
            ##################################################

            self.run_scoring()

            ##################################################
            # STEP 4
            ##################################################

            self.run_classifier()

        except Exception as e:

            logger.exception(

                "Pipeline Failed"

            )

            raise RuntimeError(

                f"Training Pipeline Failed\n{e}"

            )

        ######################################################
        # SUMMARY
        ######################################################

        total_time = round(

            time.time() - self.start_time,

            2,

        )

        logger.info("=" * 70)
        logger.info("PIPELINE FINISHED")
        logger.info("=" * 70)

        print()

        print("=" * 70)

        print("REIOS TRAINING SUMMARY")

        print("=" * 70)

        for stage, info in self.results.items():

            print(

                f"{stage:<15}"

                f"{info['status']:<10}"

                f"{info['time']} sec"

            )

        print("-" * 70)

        print(

            f"Total Time : {total_time} sec"

        )

        print("=" * 70)

        return self.results


    ##########################################################
    # PIPELINE REPORT
    ##########################################################

    def report(
        self,
    ):
        """
        Return pipeline summary dictionary.
        """

        return self.results


    ##########################################################
    # STRING REPRESENTATION
    ##########################################################

    def __repr__(
        self,
    ):

        return (

            "REIOSTrainingPipeline("

            f"steps={len(self.results)}"

            ")"

        )   
        
    ##############################################################
    # MAIN
    ##############################################################

def main():

    pipeline = REIOSTrainingPipeline()

    pipeline.run()


if __name__ == "__main__":

    main()    