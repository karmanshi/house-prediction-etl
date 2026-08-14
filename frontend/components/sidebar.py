import streamlit as st

from utils.constants import PROPERTY_TYPES, CONDITIONS, LOCATIONS


def show_sidebar(price_bounds=(0, 50000000)):

    st.sidebar.markdown("### 🔍 Filters")

    if st.sidebar.button("Clear All"):
        st.cache_data.clear()
        st.rerun()

    location = st.sidebar.selectbox("City", ["All"] + LOCATIONS)

    property_type = st.sidebar.selectbox("Property Type", ["All"] + PROPERTY_TYPES)

    condition = st.sidebar.selectbox("Condition", ["All"] + CONDITIONS)

    bedrooms = st.sidebar.slider("Bedrooms", 1, 10, (1, 10))

    bathrooms = st.sidebar.slider("Bathrooms", 1, 10, (1, 10))

    price_range = st.sidebar.slider(
        "Price Range ($)",
        min_value=int(price_bounds[0]),
        max_value = int(price_bounds[1] if price_bounds[1] is not None else 50000000),
        
        
        value=(int(price_bounds[0]), int(price_bounds[1])),
    )

    score = st.sidebar.slider("Opportunity Score", 0, 100, (0, 100))

    apply_clicked = st.sidebar.button("🔍 Apply Filters", use_container_width=True)

    return {
        "location": location,
        "property_type": property_type,
        "condition": condition,
        "price_range": price_range,
        "score": score,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "apply_clicked": apply_clicked,
    }
