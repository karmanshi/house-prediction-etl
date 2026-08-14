"""
backend/feature_transformer.py
"""

from pathlib import Path
import pandas as pd

from training.feature_manager import FeatureManager


# ROOT = Path(__file__).resolve().parent.parent

# METADATA_DIR = ROOT / "feature_metadata"


# feature_manager = FeatureManager(
#     metadata_dir=METADATA_DIR
# )


def transform_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw API input into the exact feature matrix
    used during model training.
    """

    return feature_manager.prepare_prediction_data(df)