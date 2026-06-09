from .common import ColumnNotExists

def data_transform(dataframe_extract):
    col_to_Check = [
        "Property_Type",
        "Kitchen_Type",
        "Land_Area",
        "Floor_Area",
        "Location",
        "Price",
        "View",
        "Furnishing_Status",
        "Condition",
        "Balcony (Yes/No)",
        "Num_rooms",
        "distance to nearest MRT Station",
        "distance to nearest Hospital",
        "distance to nearest School",
        "distance to nearest Bus Stand",
        "distance to nearest Airport"
    ]
    if set(col_to_Check).issubset(dataframe_extract.columns):
        print("All columns are present")
    else:
        raise ColumnNotExists()
    
    # Rename Column name
    new_column_name ={
        "Property_Type":"property_type",
        "Land_Area":"land_area",
        "Floor_Area":"floor_area",
        "Location":"location",
        "Price":"price",
        "View":"view",
        "Furnishing_Status":"is_furnished",
        "Condition":"condition",
        "Balcony (Yes/No)":"has_balcony",
        "Num_rooms":"num_rooms",
        "Kitchen_Type":"kitchen_type",
        "distance to nearest MRT Station":"dtn_mrt_station",
        "distance to nearest Hospital":"dtn_hospital",
        "distance to nearest School":"dtn_school",
        "distance to nearest Bus Stand":"dtn_bus_stand",
        "distance to nearest Airport":"dtn_airport"
    }
    dataframe_extract=dataframe_extract.rename(columns=new_column_name)

    dataframe_extract[[
        "property_type", "view", "kitchen_type", "is_furnished", "condition", "has_balcony", "location"
    ]]=dataframe_extract[[
        "property_type", "view", "kitchen_type", "is_furnished", "condition", "has_balcony", "location"
    ]].fillna('Unknown')

    dataframe_extract[[
        "dtn_airport", "dtn_bus_stand",
        "dtn_school", "dtn_hospital",
        "dtn_mrt_station", "num_rooms",
        "price", "land_area", "floor_area"
    ]]=dataframe_extract[[
        "dtn_airport", "dtn_bus_stand",
        "dtn_school", "dtn_hospital",
        "dtn_mrt_station", "num_rooms",
        "price", "land_area", "floor_area"
    ]].fillna(0.0)

    # Removing the whitespaces from values from string type columns
    string_cols = dataframe_extract.select_dtypes(include="object").columns

    for col in string_cols:
        dataframe_extract[col] = dataframe_extract[col].str.strip()

    dataframe_extract = dataframe_extract[
        (dataframe_extract["price"] > 0) &
        (dataframe_extract["land_area"] > 0) &
        (dataframe_extract["floor_area"] > 0)
    ]

    dataframe_extract["price_per_sqft"] = (
        dataframe_extract["price"] /
        dataframe_extract["floor_area"]
    )

    dataframe_extract = dataframe_extract.sort_values(
        by="price",
        ascending=False
    )

    dataframe_extract=dataframe_extract.drop_duplicates()

    # Rename Column name to originals
    new_dict= {value:key for key, value in new_column_name.items()}
    dataframe_extract=dataframe_extract.rename(columns=new_dict)

    return dataframe_extract