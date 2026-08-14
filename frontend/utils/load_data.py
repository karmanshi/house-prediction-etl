import pandas as pd
import streamlit as st

from services.api import get_summary, get_properties


@st.cache_data(ttl=60)
def load_summary():
    """Full-dataset KPIs — real totals, not capped at 100."""
    return get_summary()


@st.cache_data(ttl=60)
def load_properties(limit: int = 3000):
    """Bounded real-row sample for charts/map/table."""
    data = get_properties(limit=limit)
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    if "opportunity_score" in df.columns:
        df["Opportunity_Score"] = df["opportunity_score"]

    if "tier" in df.columns:
        df["Tier"] = df["tier"]

    if "anomaly_score" in df.columns and "anomaly_label" in df.columns:
        normal_median = df.loc[df["anomaly_label"] == 1, "anomaly_score"].median()

        def _risk(row):
            if row["anomaly_label"] == -1:
                return "High"
            return "Low" if row["anomaly_score"] >= normal_median else "Medium"

        df["Risk_Level"] = df.apply(_risk, axis=1)

    return df


def apply_filters(df, filters):
    if df.empty:
        return df

    if filters["location"] != "All" and "Location" in df.columns:
        df = df[df["Location"] == filters["location"]]

    if filters["property_type"] != "All" and "Property_Type" in df.columns:
        df = df[df["Property_Type"] == filters["property_type"]]

    if filters["condition"] != "All" and "Condition" in df.columns:
        df = df[df["Condition"] == filters["condition"]]

    if "Price" in df.columns:
        lo, hi = filters["price_range"]
        df = df[df["Price"].between(lo, hi)]

    if "Opportunity_Score" in df.columns:
        lo, hi = filters["score"]
        df = df[df["Opportunity_Score"].between(lo, hi)]

    if "Num_rooms" in df.columns:
        lo, hi = filters["bedrooms"]
        df = df[df["Num_rooms"].between(lo, hi)]

    if "Num_bathrooms" in df.columns:
        lo, hi = filters["bathrooms"]
        df = df[df["Num_bathrooms"].between(lo, hi)]

    return df
