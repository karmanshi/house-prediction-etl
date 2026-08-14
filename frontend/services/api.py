import requests
import streamlit as st

from utils.constants import BASE_API_URL as BASE_URL


def check_backend():
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def predict_property(property_data: dict):
    """Unchanged from before — sends the full 30-field payload to /predict."""
    try:
        response = requests.post(f"{BASE_URL}/predict", json=property_data, timeout=30)

        if response.status_code == 200:
            return response.json()

        st.error(f"Prediction Failed ({response.status_code})")
        st.write(response.text)
        return None

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to FastAPI backend.")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out.")
        return None
    except Exception as e:
        st.error(str(e))
        return None


@st.cache_data(ttl=60)
def get_summary():
    """
    KPI aggregates computed over the FULL dataset by the new
    /summary endpoint — not capped at 100.
    """
    try:
        response = requests.get(f"{BASE_URL}/summary", timeout=30)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


@st.cache_data(ttl=60)
def get_properties(limit: int = 3000):
    """
    Bounded real-row sample from the new /properties endpoint,
    for the map/table/scatter charts.
    """
    try:
        response = requests.get(f"{BASE_URL}/properties", params={"limit": limit}, timeout=30)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception:
        return []
