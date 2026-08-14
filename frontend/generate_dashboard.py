"""
=============================================================
REIOS Dashboard Data Generator
=============================================================

Purpose
-------
Prepare data for the Streamlit dashboard.

This module:
    • Loads feature_engineered.csv
    • Runs inference using OpportunityScorer
    • Returns a scored dataframe

No model training happens here.

=============================================================
"""

from pathlib import Path

import pandas as pd

from training.scorer import OpportunityScorer


ROOT_DIR = Path(__file__).resolve().parent.parent


class DashboardDataGenerator:
    """
    Generates dashboard-ready data.
    """

    def __init__(self):

        self.scorer = OpportunityScorer()

    def load_data(self):

        data_path = (
            ROOT_DIR
            / "data"
            / "processed"
            / "feature_engineered.csv"
        )

        if not data_path.exists():
            raise FileNotFoundError(data_path)

        return pd.read_csv(data_path)

    def generate_dashboard_data(self):

        df = self.load_data()

        scored_df = self.scorer.score_dataframe(
            df,
            include_classifier=False
        )

        return scored_df

    def save_dashboard_data(
        self,
        filename="dashboard_data.csv"
    ):

        df = self.generate_dashboard_data()

        output = (
            ROOT_DIR
            / "data"
            / "processed"
            / filename
        )

        df.to_csv(
            output,
            index=False
        )

        print(f"Dashboard data saved to {output}")

        return df


def get_dashboard_dataframe():
    """
    Used by Streamlit.
    """

    generator = DashboardDataGenerator()

    return generator.generate_dashboard_data()


if __name__ == "__main__":

    generator = DashboardDataGenerator()

    df = generator.generate_dashboard_data()

    print(df.head())

    # Optional:
    # generator.save_dashboard_data()