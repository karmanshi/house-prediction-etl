"""
=============================================================
REIOS - FastAPI Backend Tests
=============================================================

Purpose
-------
Test backend API endpoints.

Tests:
------
1. Application starts
2. Health check
3. Root endpoint
4. API availability

=============================================================
"""


from fastapi.testclient import TestClient

import pytest


##############################################################
# IMPORT FASTAPI APP
##############################################################

from backend.main import app



##############################################################
# CREATE TEST CLIENT
##############################################################

client = TestClient(app)



##############################################################
# TEST APP CREATION
##############################################################

def test_api_app_created():

    """
    Verify FastAPI application
    loads correctly.
    """

    assert app is not None



##############################################################
# TEST ROOT ENDPOINT
##############################################################

def test_root_endpoint():

    """
    Test:

    GET /

    Expected:
    API should respond.
    """

    response = client.get("/")


    assert response.status_code == 200


    data = response.json()


    assert isinstance(

        data,

        dict

    )



##############################################################
# TEST HEALTH ENDPOINT
##############################################################

def test_health_endpoint():

    """
    Test:

    GET /health

    Used by:
    - Docker
    - Kubernetes
    - Load balancers

    """

    response = client.get(

        "/health"

    )


    assert response.status_code == 200


    data = response.json()


    assert "status" in data



##############################################################
# TEST HEALTH STATUS
##############################################################

def test_health_status():

    """
    Health endpoint should
    return healthy state.
    """

    response = client.get(

        "/health"

    )


    data = response.json()


    assert data["status"] in [

        "healthy",

        "ok",

        "running"

    ]
    ##############################################################
# SAMPLE PROPERTY REQUEST
##############################################################

sample_property = {

    "Land_Area": 1500,

    "Floor_Area": 1200,

    "Num_rooms": 3,

    "Num_bathrooms": 2,

    "Maintenance_Fees": 5000,

    "Latitude": 1.3521,

    "Longitude": 103.8198,

    "Crimerate": 0.02,

    "dist_MRT": 500,

    "dist_Hospital": 1000,

    "dist_School": 700,

    "dist_BusStand": 300,

    "dist_Airport": 15000,

    "Property_Type": "Apartment",

    "Condition": "New",

    "Location": "Central",

    "Furnishing_Status": "Fully Furnished",

    "View": "City",

    "Bar": 0,

    "Elevator": 1,

    "Garden": 1,

    "Gym": 1,

    "Parking": 1,

    "Swimming Pool": 1,

    "WiFi": 1

}



##############################################################
# TEST PREDICTION ENDPOINT EXISTS
##############################################################

def test_prediction_endpoint_exists():

    """
    Verify prediction route exists.

    """

    response = client.post(

        "/predict",

        json=sample_property

    )


    assert response.status_code != 404



##############################################################
# TEST SUCCESSFUL PREDICTION
##############################################################

def test_prediction_success():

    """
    Prediction API should return
    investment analysis.
    """

    response = client.post(

        "/predict",

        json=sample_property

    )


    assert response.status_code == 200


    data = response.json()


    assert isinstance(

        data,

        dict

    )



##############################################################
# TEST PREDICTED PRICE OUTPUT
##############################################################

def test_prediction_contains_price():

    """
    Hedonic model output.
    """

    response = client.post(

        "/predict",

        json=sample_property

    )


    data = response.json()


    assert (

        "predicted_price"

        in data

    )


    assert isinstance(

        data["predicted_price"],

        (int,float)

    )



##############################################################
# TEST OPPORTUNITY SCORE OUTPUT
##############################################################

def test_prediction_contains_score():

    """
    Composite scorer output.
    """

    response = client.post(

        "/predict",

        json=sample_property

    )


    data = response.json()


    assert (

        "opportunity_score"

        in data

    )


    assert 0 <= data[

        "opportunity_score"

    ] <= 100



##############################################################
# TEST TIER OUTPUT
##############################################################

def test_prediction_contains_tier():

    """
    Investment category.
    """

    response = client.post(

        "/predict",

        json=sample_property

    )


    data = response.json()


    assert "tier" in data


    assert data["tier"] in [

        "Low",

        "Fair",

        "Good",

        "Excellent"

    ]
