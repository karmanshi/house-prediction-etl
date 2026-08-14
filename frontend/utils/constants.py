BASE_API_URL = "http://127.0.0.1:8000"

APP_TITLE = "REIOS Dashboard"
APP_SUBTITLE = "Real Estate Investment Opportunity Scorer"

# Real tier names from the trained classifier — NOT renamed to
# "Strong Buy / Buy / Hold / Avoid" to avoid implying the model
# outputs something it doesn't. Colors chosen to mirror that
# green -> red spectrum from best to worst.
TIER_COLORS = {
    "Excellent": "#16A34A",  # green
    "Good": "#F59E0B",       # amber
    "Fair": "#3B82F6",       # blue
    "Low": "#DC2626",        # red
    "Total Properties": "#6366F1",
    "Average Score": "#8B5CF6",
}
TIER_ORDER = ["Excellent", "Good", "Fair", "Low"]

RISK_COLORS = {
    "Low": "#16A34A",
    "Medium": "#F59E0B",
    "High": "#DC2626",
}

PROPERTY_TYPES = ["Apartment", "Bungalow", "Condo", "Farmhouse", "Penthouse", "Villa"]
CONDITIONS = ["New", "Old", "Renovated"]
KITCHEN_TYPES = ["Modular", "Normal", "Semi Modular"]
VIEW_TYPES = ["City View", "Park Facing", "Sea Facing", "Unknown"]
FURNISHING_STATUS = ["Fully Furnished", "Semi Furnished", "Unfurnished"]
LOCATIONS = [
    "Boston", "Chicago", "Denver", "Houston", "Los Angeles",
    "Miami", "New York", "Phoenix", "San Francisco", "Seattle",
]
