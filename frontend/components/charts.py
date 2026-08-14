import streamlit as st
import pandas as pd
import plotly.express as px

from utils.constants import TIER_COLORS, TIER_ORDER, RISK_COLORS
import components.charts

def plot_tier_donut(summary):
    """Main donut — matches the reference image's structure, using
    the model's real tier names (Excellent/Good/Fair/Low) instead of
    invented labels the model doesn't actually output."""
    tier_counts = summary.get("tier_counts", {}) if summary else {}
    if not tier_counts:
        st.warning("No tier data available.")
        return

    labels = [t for t in TIER_ORDER if t in tier_counts]
    values = [tier_counts[t] for t in labels]
    colors = [TIER_COLORS[t] for t in labels]

    fig = px.pie(
        names=labels,
        values=values,
        hole=0.55,
        color=labels,
        color_discrete_map=TIER_COLORS,
    )
    fig.update_traces(textinfo="label+percent")
    fig.update_layout(height=380, showlegend=True, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)


def plot_risk_donut(df):
    if "Risk_Level" not in df.columns or df.empty:
        st.warning("Risk Level data not available.")
        return

    counts = df["Risk_Level"].value_counts().reset_index()
    counts.columns = ["Risk_Level", "Count"]

    fig = px.pie(
        counts, names="Risk_Level", values="Count", hole=0.55,
        color="Risk_Level", color_discrete_map=RISK_COLORS,
    )
    fig.update_traces(textinfo="label+percent")
    fig.update_layout(height=340, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)


def plot_properties_by_city(summary):
    city_counts = summary.get("properties_by_city", {}) if summary else {}
    if not city_counts:
        st.warning("City data not available.")
        return

    data = pd.DataFrame(list(city_counts.items()), columns=["City", "Properties"])
    data = data.sort_values("Properties", ascending=True)

    fig = px.bar(data, x="Properties", y="City", orientation="h", template="plotly_white")
    fig.update_layout(height=380, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)


def plot_actual_vs_predicted(df):
    if "Price" not in df.columns or "predicted_price" not in df.columns or df.empty:
        st.warning("Price data not available.")
        return



    fig = px.scatter(
        df, x="Price", y="predicted_price", opacity=0.4,
        template="plotly_white",
        labels={"Price": "Actual Price ($)", "predicted_price": "Predicted Price ($S)"},
    )
    max_val = max(df["Price"].max(), df["predicted_price"].max())
    fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color="green", dash="dash"))
    fig.update_layout(height=380, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)


def plot_score_histogram(df):
    if "Opportunity_Score" not in df.columns or df.empty:
        st.warning("Opportunity Score data not available.")
        return

    fig = px.histogram(df, x="Opportunity_Score", nbins=25, template="plotly_white")
    fig.update_layout(height=340, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)


def plot_avg_score_by_property_type(summary):
    scores = summary.get("avg_score_by_property_type", {}) if summary else {}
    if not scores:
        st.warning("Property type data not available.")
        return

    data = pd.DataFrame(list(scores.items()), columns=["Property_Type", "Avg_Score"])
    data = data.sort_values("Avg_Score", ascending=True)

    fig = px.bar(data, x="Avg_Score", y="Property_Type", orientation="h", template="plotly_white")
    fig.update_layout(height=340, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)


# components/charts.py — plot_property_map()

def plot_property_map(df):
    if "Latitude" not in df.columns or "Longitude" not in df.columns or df.empty:
        st.warning("Location data not available.")
        return

    center_lat = df["Latitude"].mean()
    center_lon = df["Longitude"].mean()
    
    fig = px.scatter_mapbox(
        df, lat="Latitude", lon="Longitude",
        color="Tier" if "Tier" in df.columns else None,
        color_discrete_map=TIER_COLORS,
        hover_data=["Price", "opportunity_score"] if "opportunity_score" in df.columns else None,
        zoom=3, height=420,
    )
    fig.update_layout(mapbox_style="open-street-map", margin=dict(t=10, b=10, l=0, r=0))

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"scrollZoom": True},   # ADDED — enables mouse-wheel/pinch zoom on the map
    )

# def render_tier_card(label, count, color):
#     """Colored KPI card — light background tint of `color`, bold text in `color`."""
    
#     st.markdown(
#         f"""
#         <div style="
#             background-color:{color}22;
#             border-left:5px solid {color};
#             border-radius:10px;
#             padding:14px 16px;
#             text-align:center;
#         ">
#             <div style="font-size:14px; color:{color}; font-weight:600;">{label}</div>
#             <div style="font-size:26px; color:{color}; font-weight:bold;">{count:,}</div>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )

# 
def render_tier_card(label, value, color):
    display_value = f"{value:,}" if isinstance(value, (int, float)) else value

    st.markdown(
        f"""
        <div style="
            background-color:{color}22;
            border-left:5px solid {color};
            border-radius:10px;
            padding:14px 12px;
            text-align:center;
            overflow:hidden;
        ">
            <div style="
                font-size:14spx;
                color:{color};
                font-weight:700;
                white-space:nowrap;
                overflow:hidden;
                text-overflow:ellipsis;
            ">{label}</div>
            <div style="
                font-size:28px;
                color:{color};
                font-weight:900;
                white-space:nowrap;
            ">{display_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )