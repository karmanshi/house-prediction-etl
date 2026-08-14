"""
=============================================================
REIOS Feature Metadata Generator
=============================================================

Creates metadata required for:
- Backend prediction
- Feature alignment
- Model inference

Generates:
    feature_order.pkl
    feature_columns.pkl
    numeric_columns.pkl
    categorical_levels.pkl
    property_price.pkl
    location_price.pkl
    training_medians.pkl
    training_modes.pkl

=============================================================
"""


from pathlib import Path
import sys
import joblib
import pandas as pd


#############################################################
# ADD PROJECT ROOT TO PYTHON PATH
#############################################################

ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT_DIR))


from training.feature_manager import FeatureManager



#############################################################
# PATHS
#############################################################

DATA_DIR = ROOT_DIR / "data" / "processed"

FEATURE_METADATA_DIR = (
    ROOT_DIR / "feature_metadata"
)

FEATURE_METADATA_DIR.mkdir(
    exist_ok=True
)



#############################################################
# LOAD DATA
#############################################################

print("Loading dataset...")


df = pd.read_csv(
    DATA_DIR / "feature_engineered.csv"
)


df_copy = df.copy()



#############################################################
# HEDONIC FEATURE ORDER
#############################################################

feature_columns = (
    FeatureManager
    .get_hedonic_features(df)
)


joblib.dump(
    feature_columns,
    FEATURE_METADATA_DIR /
    "feature_columns.pkl"
)


joblib.dump(
    feature_columns,
    FEATURE_METADATA_DIR /
    "feature_order.pkl"
)


print(
    len(feature_columns),
    "hedonic features saved"
)



#############################################################
# NUMERIC COLUMNS
#############################################################

numeric_columns = list(

    df.select_dtypes(
        include="number"
    ).columns

)


joblib.dump(
    numeric_columns,
    FEATURE_METADATA_DIR /
    "numeric_columns.pkl"
)


print(
    len(numeric_columns),
    "numeric columns saved"
)



#############################################################
# CATEGORICAL LEVELS FROM ONE-HOT COLUMNS
#############################################################

categorical_levels = {}


prefix_mapping = {

    "Location":
        "Location_",

    "Property_Type":
        "Property_Type_",

    "Condition":
        "Condition_",

    "Kitchen_Type":
        "Kitchen_Type_",

    "View":
        "View_",

    "Furnishing_Status":
        "Furnishing_Status_"

}


for name, prefix in prefix_mapping.items():

    levels = [

        col.replace(prefix, "")

        for col in df.columns

        if col.startswith(prefix)

    ]

    categorical_levels[name] = sorted(levels)



joblib.dump(

    categorical_levels,

    FEATURE_METADATA_DIR /
    "categorical_levels.pkl"

)


print(
    "Categorical levels saved"
)



#############################################################
# LOCATION MEDIAN PRICE
#############################################################

location_cols = [

    col

    for col in df.columns

    if col.startswith("Location_")

]


df_temp = df.copy()


df_temp["Location"] = (

    df_temp[location_cols]

    .idxmax(axis=1)

    .str.replace(
        "Location_",
        "",
        regex=False
    )

)


location_price = (

    df_temp

    .groupby("Location")["Price"]

    .median()

    .to_dict()

)


joblib.dump(

    location_price,

    FEATURE_METADATA_DIR /
    "location_price.pkl"

)


print(
    "Location prices saved"
)


#############################################################
# PROPERTY TYPE MEDIAN PRICE
#############################################################

property_cols = [

    col

    for col in df.columns

    if col.startswith("Property_Type_")

]


df_temp["Property_Type"] = (

    df_temp[property_cols]

    .idxmax(axis=1)

    .str.replace(
        "Property_Type_",
        "",
        regex=False
    )

)



property_price = (

    df_temp

    .groupby("Property_Type")["Price"]

    .median()

    .to_dict()

)


joblib.dump(

    property_price,

    FEATURE_METADATA_DIR /
    "property_price.pkl"

)


print(
    "Property prices saved"
)

#############################################################
# TRAINING MEDIANS
#############################################################

training_medians = {}


for col in df.columns:

    if pd.api.types.is_numeric_dtype(
        df[col]
    ):

        training_medians[col] = (
            df[col]
            .median()
        )



joblib.dump(
    training_medians,
    FEATURE_METADATA_DIR /
    "training_medians.pkl"
)



print(
    len(training_medians),
    "medians saved"
)

#############################################################
# SAVE NORMALIZATION STATS
#############################################################

normalization_stats = {}


for col in df.select_dtypes(
    include="number"
).columns:

    normalization_stats[col] = {

        "mean": float(df[col].mean()),

        "std": float(df[col].std()),

        "min": float(df[col].min()),

        "max": float(df[col].max()),

        "p1": float(df[col].quantile(0.01)),

        "p99": float(df[col].quantile(0.99))

    }


#############################################################
# ADD INFERENCE GENERATED FEATURES
#############################################################

# These features are created after model prediction

for col in [
    "residual_pct",
    "anomaly_score"
]:

    normalization_stats[col] = {

        "mean":0.0,

        "std":1.0,

        "min":-1.0,

        "max":1.0,

        "p1":-1.0,

        "p99":1.0

    }



joblib.dump(

    normalization_stats,

    FEATURE_METADATA_DIR /
    "normalization_stats.pkl"

)


print(
    len(normalization_stats),
    "normalization stats saved"
)

#############################################################
# TRAINING MODES
#############################################################

training_modes = {}


for col in df.columns:

    if not pd.api.types.is_numeric_dtype(
        df[col]
    ):

        mode = df[col].mode()

        if len(mode):

            training_modes[col] = (
                mode.iloc[0]
            )



joblib.dump(
    training_modes,
    FEATURE_METADATA_DIR /
    "training_modes.pkl"
)


print(
    len(training_modes),
    "modes saved"
)



#############################################################
# VERIFY
#############################################################

print("\nMetadata Files:")


for file in FEATURE_METADATA_DIR.glob(
    "*.pkl"
):

    print(
        file.name
    )


print("\nFeature metadata generation completed.")