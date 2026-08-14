"""
=============================================================
REIOS - Model Validation Tests
=============================================================

Purpose
-------
Validate trained ML models before deployment.

Tests:
------
1. Model files exist
2. Models load correctly
3. Loaded objects are valid
4. Required artifacts exist

=============================================================
"""


from pathlib import Path

import joblib
import pytest


##############################################################
# PROJECT PATH
##############################################################

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT_DIR / "models"



##############################################################
# TEST MODEL DIRECTORY
##############################################################

def test_model_directory_exists():
    """
    Ensure models directory exists.
    """

    assert MODEL_DIR.exists(), (

        "Models directory missing"

    )



##############################################################
# TEST REQUIRED MODEL FILES
##############################################################

def test_required_model_files_exist(
    required_model_files,
):
    """
    Verify all trained artifacts exist.

    Generated after training:
    
    train_hedonic.py
    train_anomaly.py
    train_classifier.py
    """

    missing = [

        str(path)

        for path in required_model_files

        if not path.exists()

    ]


    assert missing == [], (

        f"Missing model files:\n{missing}"

    )



##############################################################
# TEST HEDONIC MODEL LOAD
##############################################################

def test_load_hedonic_model(
    hedonic_model,
):
    """
    LightGBM regression model
    should load successfully.
    """

    assert hedonic_model is not None



##############################################################
# TEST ANOMALY MODEL LOAD
##############################################################

def test_load_anomaly_model(
    anomaly_model,
):
    """
    Isolation Forest should load.
    """

    assert anomaly_model is not None



##############################################################
# TEST CLASSIFIER MODEL LOAD
##############################################################

def test_load_classifier_model(
    classifier_model,
):
    """
    LightGBM classifier should load.
    """

    assert classifier_model is not None



##############################################################
# TEST ENCODERS LOAD
##############################################################

def test_load_encoders(
    tier_encoder,
    
):
    """
    Encoders should be available
    for prediction pipeline.
    """

    assert tier_encoder is not None

    
    
##############################################################
# TEST HEDONIC MODEL PREDICTION
##############################################################

def test_hedonic_prediction(
    hedonic_model,
    sample_hedonic_input,
):
    """
    Test LightGBM regression inference.

    Expected:
    - one prediction
    - numeric output
    - no NaN
    """

    prediction = hedonic_model.predict(

        sample_hedonic_input

    )


    assert len(prediction) == 1


    assert prediction[0] == prediction[0], (
        "Prediction is NaN"
    )


    assert isinstance(

        prediction[0],

        float

    )



##############################################################
# TEST HEDONIC PRICE RANGE
##############################################################

def test_hedonic_prediction_range(
    hedonic_model,
    sample_hedonic_input,
):
    """
    Verify predicted log price is reasonable.

    Prevents broken models producing
    extreme values.
    """

    prediction = hedonic_model.predict(

        sample_hedonic_input

    )[0]


    assert prediction > 0


    assert prediction < 30



##############################################################
# TEST ANOMALY MODEL PREDICTION
##############################################################

def test_anomaly_prediction(
    anomaly_model,
    sample_anomaly_input,
):
    """
    Test Isolation Forest inference.

    Output:
        1  -> normal
       -1  -> anomaly
    """

    prediction = anomaly_model.predict(

        sample_anomaly_input

    )


    assert len(prediction) == 1


    assert prediction[0] in [

        1,

        -1

    ]



##############################################################
# TEST ANOMALY SCORE
##############################################################

def test_anomaly_score(
    anomaly_model,
    sample_anomaly_input,
):
    """
    Isolation Forest should produce
    anomaly scores.
    """

    score = anomaly_model.score_samples(

        sample_anomaly_input

    )


    assert len(score) == 1


    assert score[0] == score[0]



##############################################################
# TEST CLASSIFIER PREDICTION
##############################################################

def test_classifier_prediction(
    classifier_model,
    sample_classifier_input,
):
    """
    Test investment tier classifier.
    """

    prediction = classifier_model.predict(

        sample_classifier_input

    )


    assert len(prediction) == 1


    assert prediction[0] == prediction[0]



##############################################################
# TEST CLASSIFIER PROBABILITY
##############################################################

def test_classifier_probability(
    classifier_model,
    sample_classifier_input,
):
    """
    Classifier should provide probabilities.

    Used later by API/dashboard confidence.
    """

    if hasattr(

        classifier_model,

        "predict_proba"

    ):

        probabilities = (

            classifier_model

            .predict_proba(

                sample_classifier_input

            )

        )


        assert probabilities.shape[0] == 1


        assert (

            probabilities.sum()

            == pytest.approx(1.0)

        )
        
        
##############################################################
# TEST HEDONIC FEATURE COUNT
##############################################################

def test_hedonic_feature_count(
    hedonic_model,
    hedonic_features,
):
    """
    Ensure inference feature count matches
    training feature count.

    Prevents:
    Feature mismatch errors.
    """

    expected_features = len(
        hedonic_features
    )


    model_features = (

        hedonic_model

        .n_features_in_

    )


    assert model_features == expected_features, (

        f"Model expects {model_features} features "
        f"but FeatureManager provides "
        f"{expected_features}"

    )



##############################################################
# TEST ANOMALY FEATURE COUNT
##############################################################

def test_anomaly_feature_count(
    anomaly_model,
    anomaly_features,
):
    """
    Validate Isolation Forest input size.
    """

    expected_features = len(
        anomaly_features
    )


    model_features = (

        anomaly_model

        .n_features_in_

    )


    assert model_features == expected_features



##############################################################
# TEST CLASSIFIER FEATURE COUNT
##############################################################

def test_classifier_feature_count(
    classifier_model,
    classifier_features,
):
    """
    Validate classifier feature count.
    """

    expected_features = len(

        classifier_features

    )


    model_features = (

        classifier_model

        .n_features_in_

    )


    assert model_features == expected_features



##############################################################
# TEST BATCH HEDONIC PREDICTION
##############################################################

def test_batch_hedonic_prediction(
    hedonic_model,
    engineered_df,
    hedonic_features,
):
    """
    Test predicting multiple properties.

    Simulates dashboard scoring.
    """

    X = (

        engineered_df

        [hedonic_features]

        .fillna(0)

        .head(10)

    )


    predictions = (

        hedonic_model

        .predict(X)

    )


    assert len(predictions) == 10


    assert all(

        np.isfinite(predictions)

    )



##############################################################
# TEST BATCH ANOMALY PREDICTION
##############################################################

def test_batch_anomaly_prediction(
    anomaly_model,
    engineered_df,
    anomaly_features,
):
    """
    Test anomaly detection
    for multiple properties.
    """

    X = (

        engineered_df

        [anomaly_features]

        .fillna(0)

        .head(20)

    )


    labels = anomaly_model.predict(X)


    assert len(labels) == 20


    assert set(labels).issubset(

        {

            1,

            -1

        }

    )



##############################################################
# TEST BATCH CLASSIFICATION
##############################################################

def test_batch_classifier_prediction(
    classifier_model,
    engineered_df,
    classifier_features,
):
    """
    Test tier classification
    for multiple properties.
    """

    existing_features = [

        c

        for c in classifier_features

        if c in engineered_df.columns

    ]


    X = (

        engineered_df

        [existing_features]

        .fillna(0)

        .head(10)

    )


    predictions = classifier_model.predict(X)


    assert len(predictions) == 10



##############################################################
# TEST MODEL REPRODUCIBILITY
##############################################################

def test_model_prediction_consistency(
    hedonic_model,
    sample_hedonic_input,
):
    """
    Same input should produce
    same prediction.

    Checks deterministic behaviour.
    """

    pred1 = hedonic_model.predict(

        sample_hedonic_input

    )


    pred2 = hedonic_model.predict(

        sample_hedonic_input

    )


    assert pred1 == pytest.approx(

        pred2

    )
    
    
