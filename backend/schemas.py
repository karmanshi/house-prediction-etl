"""
=============================================================
REIOS - API Schemas
=============================================================

Defines request and response models for FastAPI.

=============================================================
"""

from pydantic import BaseModel


class PropertyInput(BaseModel):

    # Basic
    Location: str
    City: str
    Price: float

    # Property
    Property_Type: str
    Condition: str
    Kitchen_Type: str
    View: str
    Furnishing_Status: str

    # Area
    Land_Area: float
    Floor_Area: float

    # Rooms
    Num_rooms: int
    Num_bathrooms: int

    # Financial
    Maintenance_Fees: float

    # Coordinates
    Latitude: float
    Longitude: float

    # Distances
    dist_MRT: float
    dist_Hospital: float
    dist_School: float
    dist_BusStand: float
    dist_Airport: float

    # Crime
    Crimerate: float

    # Amenities
    Bar: int
    Elevator: int
    Garden: int
    Gym: int
    Parking: int
    Swimming_Pool: int
    WiFi: int
    Air_Conditioning: int
    Heating: int
    Balcony: int


class PredictionResponse(BaseModel):
    """
    API prediction response.
    """

    predicted_price: float
    residual_pct: float
    opportunity_score: float
    anomaly_score: float
    anomaly_label: int
    tier: str
    percentile: float