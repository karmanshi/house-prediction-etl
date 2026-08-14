import pandas as pd

from training.scorer import OpportunityScorer


scorer = OpportunityScorer()
scorer.load_models(include_classifier=True)


def predict_property(property_data: dict) -> dict:
    """
    Score a single property and return only API contract fields.
    """
    df = pd.DataFrame([property_data])
    scored = scorer.score_dataframe(df)
    row = scored.iloc[0]

    return {
        "predicted_price": float(row["predicted_price"]),
        "residual_pct": float(row["residual_pct"]) if pd.notna(row["residual_pct"]) else 0.0,
        "opportunity_score": float(row["opportunity_score"]),
        "anomaly_score": float(row["anomaly_score"]),
        "anomaly_label": int(row["anomaly_label"]),
        "tier": str(row["tier"]),
        "percentile": float(row["percentile"]),
    }
