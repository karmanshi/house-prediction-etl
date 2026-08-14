import streamlit as st


def property_form():
    """
    Property Input Form
    Returns a dictionary that matches the FastAPI request schema.
    """

    st.subheader("🏠 Property Information")

    col1, col2 = st.columns(2)

    with col1:

        # location = st.text_input(
        #     "Location",
        #     value="Whitefield",
        # )

        location = st.selectbox(
            "Location",
            [
                'San Francisco', 
                'New York', 
                'Los Angeles', 
                'Boston', 
                'Chicago',
                'Seattle',
                'Miami', 
                'Denver', 
                'Houston', 
                'Phoenix'
            ],
        )

        property_type = st.selectbox(
            "Property Type",
            [
                "Apartment", 
                "Bungalow", 
                "Condo", 
                "Farmhouse", 
                "Penthouse", 
                "Villa"
            ],
        )

        condition = st.selectbox(
            "Condition",
            [
                "New", "Old", "Renovated"
            ],
        )

        kitchen_type = st.selectbox(
            "Kitchen Type",
            [
                "Modular", 
                "Normal", 
                "Semi Modular"
            ],
        )

        furnishing = st.selectbox(
            "Furnishing Status",
            [
                "Fully Furnished",
                "Semi Furnished",
                "Unfurnished",
            ],
        )

    with col2:

        view = st.selectbox(
            "View",
            [
                "City View", 
                "Park Facing", 
                "Sea Facing", 
                "Unknown"
            ],
        )

        price = st.number_input(
            "Price (₹)",
            min_value=100000,
            value=6500000,
            step=100000,
        )

        land_area = st.number_input(
            "Land Area (sqft)",
            min_value=100,
            value=1200,
        )

        floor_area = st.number_input(
            "Floor Area (sqft)",
            min_value=100,
            value=1000,
        )

        rooms = st.number_input(
            "Number of Rooms",
            min_value=1,
            max_value=20,
            value=3,
        )

        bathrooms = st.number_input(
            "Number of Bathrooms",
            min_value=1,
            max_value=10,
            value=2,
        )

        maintenance = st.number_input(
            "Maintenance Fees",
            min_value=0,
            value=3000,
        )

    st.markdown("---")

    st.subheader("📍 Location Details")

    col1, col2 = st.columns(2)

    with col1:

        latitude = st.number_input(
            "Latitude",
            value=12.9716,
            format="%.6f",
        )

        longitude = st.number_input(
            "Longitude",
            value=77.5946,
            format="%.6f",
        )

        dist_mrt = st.number_input(
            "Distance to MRT (km)",
            value=2.0,
        )

        dist_hospital = st.number_input(
            "Distance to Hospital (km)",
            value=1.0,
        )

        dist_school = st.number_input(
            "Distance to School (km)",
            value=1.5,
        )

    with col2:

        dist_bus = st.number_input(
            "Distance to Bus Stand (km)",
            value=0.5,
        )

        dist_airport = st.number_input(
            "Distance to Airport (km)",
            value=25.0,
        )

        crime = st.slider(
            "Crime Rate",
            0.0,
            10.0,
            2.5,
        )

    st.markdown("---")

    st.subheader("🏊 Amenities")

    c1, c2, c3 = st.columns(3)

    with c1:
        bar = st.checkbox("Bar")
        garden = st.checkbox("Garden")
        parking = st.checkbox("Parking")

    with c2:
        elevator = st.checkbox("Elevator")
        gym = st.checkbox("Gym")
        swimming_pool = st.checkbox("Swimming Pool")
        air_conditioning = st.checkbox("Air Conditioning")

    with c3:
        wifi = st.checkbox("WiFi")
        heating = st.checkbox("Heating")     
        balcony = st.checkbox("Balcony")

    return {
        "Location": location,
        "City": location,
        "Price": price,
        "Property_Type": property_type,
        "Condition": condition,
        "Kitchen_Type": kitchen_type,
        "View": view,
        "Furnishing_Status": furnishing,
        "Land_Area": land_area,
        "Floor_Area": floor_area,
        "Num_rooms": rooms,
        "Num_bathrooms": bathrooms,
        "Maintenance_Fees": maintenance,
        "Latitude": latitude,
        "Longitude": longitude,
        "dist_MRT": dist_mrt,
        "dist_Hospital": dist_hospital,
        "dist_School": dist_school,
        "dist_BusStand": dist_bus,
        "dist_Airport": dist_airport,
        "Crimerate": crime,
        "Bar": int(bar),
        "Elevator": int(elevator),
        "Garden": int(garden),
        "Gym": int(gym),
        "Parking": int(parking),
        "Swimming_Pool": int(swimming_pool),
        "WiFi": int(wifi),
        "Air_Conditioning": int(air_conditioning),   
        "Heating": int(heating),                     
        "Balcony": int(balcony),
    }