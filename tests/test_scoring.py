"""
=============================================================
REIOS - Composite Scoring Tests
=============================================================

Purpose
-------
Validate investment opportunity scoring logic.

Tests:
------
1. Scoring utilities
2. Normalization
3. Score ranges
4. Weight validation

=============================================================
"""


import numpy as np
import pandas as pd
import pytest


##############################################################
# IMPORT SCORER MODULE
##############################################################
from training.scorer import OpportunityScorer

scorer = OpportunityScorer()

minmax_clip = scorer.minmax_clip
WEIGHTS = scorer.weights

##############################################################
# TEST MINMAX CLIP FUNCTION
##############################################################

def test_minmax_clip_basic(scorer):

    """
    Test normalization.

    Expected output:
    
    smallest value -> 0
    largest value  -> 1

    """

    series = pd.Series(
        [
            10,
            20,
            30,
            40,
            50
        ]
    )


    result = scorer.minmax_clip(series)


    assert result.min() == pytest.approx(
        0.0
    )


    assert result.max() == pytest.approx(
        1.0
    )



##############################################################
# TEST MINMAX CLIP RANGE
##############################################################

def test_minmax_clip_range():

    """
    Normalized values should
    always stay between 0 and 1.
    """

    data = pd.Series(

        np.random.randn(100)

    )


    result = scorer.minmax_clip(data)


    assert (

        result >= 0

    ).all()


    assert (

        result <= 1

    ).all()



##############################################################
# TEST CONSTANT SERIES
##############################################################

def test_minmax_clip_constant_values():

    """
    Edge case:

    When all values are identical,
    range becomes zero.

    Function should return 0.5.
    """

    series = pd.Series(

        [

            100,

            100,

            100,

            100

        ]

    )


    result = scorer.minmax_clip(series)


    assert (

        result == 0.5

    ).all()



##############################################################
# TEST SCORING WEIGHTS EXIST
##############################################################

def test_scoring_weights_exist():

    """
    Validate composite scoring weights.
    """

    required_weights = [

        "value_gap_score",

        "growth_score",

        "access_score",

        "safety_score",

        "anomaly_score_n",

    ]


    for weight in required_weights:

        assert weight in scorer.weights



##############################################################
# TEST WEIGHT SUM
##############################################################

def test_scoring_weight_sum():

    """
    Total weights should not exceed 1.

    Current design:

    value gap = 35%
    growth    = 20%
    access    = 15%
    safety    = 12%
    anomaly   = 5%

    """

    total = sum(

        scorer.weights.values()

    )


    assert total <= 1.0



##############################################################
# TEST SCORE OUTPUT RANGE
##############################################################

def test_normalized_score_range():

    """
    Test expected normalized score range.
    """

    sample = pd.Series(

        [

            0.1,

            0.5,

            0.9

        ]

    )


    normalized = minmax_clip(

        sample

    )


    assert normalized.between(

        0,

        1

    ).all()
    
##############################################################
# TEST VALUE GAP SCORE
##############################################################

def test_value_gap_score_calculation():
    """
    Value gap score should reward
    underpriced properties.

    Formula:

    value_gap_score =
        minmax_clip(-residual_pct)

    Negative residual:
        actual < predicted
        property is undervalued
        higher score

    """

    residuals = pd.Series(
        [
            -20,
            -10,
            0,
            10,
            20
        ]
    )


    value_scores = scorer.minmax_clip(

        -residuals

    )


    # Highest score should belong
    # to most undervalued property

    assert (

        value_scores.iloc[0]

        >

        value_scores.iloc[-1]

    )



##############################################################
# TEST GROWTH SCORE
##############################################################

def test_growth_score_calculation():
    """
    Growth score should increase
    with neighbourhood growth.

    Formula:

    growth_score =
        minmax_clip(loc_price_median)

    """

    growth = pd.Series(

        [

            100,

            200,

            300,

            400,

            500

        ]

    )


    scores = scorer.minmax_clip(

        growth

    )


    assert scores.iloc[-1] > scores.iloc[0]


    assert scores.between(

        0,

        1

    ).all()



##############################################################
# TEST ACCESS SCORE
##############################################################

def test_access_score_calculation():
    """
    Lower accessibility distance penalty
    should produce higher score.

    Formula:

    access_score =
        minmax_clip(-accessibility)

    """

    accessibility = pd.Series(

        [

            100,

            500,

            1000,

            2000

        ]

    )


    scores = scorer.minmax_clip(

        -accessibility

    )


    # Closest property wins

    assert (

        scores.iloc[0]

        >

        scores.iloc[-1]

    )



##############################################################
# TEST SAFETY SCORE
##############################################################

def test_safety_score_calculation():
    """
    Lower crime rate should
    produce higher safety score.

    Formula:

    safety_score =
        minmax_clip(-Crimerate)

    """

    crime = pd.Series(

        [

            0.01,

            0.05,

            0.10,

            0.20

        ]

    )


    scores = scorer.minmax_clip(

        -crime

    )


    assert (

        scores.iloc[0]

        >

        scores.iloc[-1]

    )



##############################################################
# TEST ANOMALY NORMALIZATION
##############################################################

def test_anomaly_score_normalization():

    """
    Check anomaly score inversion.

    Isolation Forest:

    lower score =
        more anomalous

    Formula:

    anomaly_score_n =
        minmax_clip(-anomaly_score)

    """

    anomaly_scores = pd.Series(

        [

            -0.5,

            -0.2,

            0.1

        ]

    )


    normalized = scorer.minmax_clip(

        -anomaly_scores

    )


    assert normalized.between(

        0,

        1

    ).all()



##############################################################
# TEST OPPORTUNITY SCORE FORMULA
##############################################################

def test_opportunity_score_formula():

    """
    Verify weighted scoring.

    Formula:

    opportunity_score =
        (
        value_gap*0.35 +
        growth*0.20 +
        access*0.15 +
        safety*0.12 +
        anomaly*0.05
        )
        *100

    """

    scores = {

        "value_gap_score":1.0,

        "growth_score":1.0,

        "access_score":1.0,

        "safety_score":1.0,

        "anomaly_score_n":1.0,

    }


    expected = (

        0.35 +

        0.20 +

        0.15 +

        0.12 +

        0.05

    ) * 100


    calculated = sum(

        scores[key] * scorer.weights[key]

        for key in scores

    ) * 100


    assert calculated == pytest.approx(

        expected

    )
    
##############################################################
# IMPORT COMPLETE SCORER
##############################################################

from training.scorer import OpportunityScorer

scorer = OpportunityScorer()
##############################################################
# TEST SCORER OUTPUT COLUMNS
##############################################################

def test_scorer_output_columns(
    scorer,
    engineered_df,
):
    scored_df = scorer.score_dataframe(
        engineered_df.head(20),
        include_classifier=False,
    )


    required_columns = [

        "predicted_price",

        "residual_pct",

        "anomaly_score",

        "anomaly_label",

        "value_gap_score",

        "growth_score",

        "access_score",

        "safety_score",

        "anomaly_score_n",

        "opportunity_score",

        "tier",

        "percentile",

    ]


    for column in required_columns:

        assert column in scored_df.columns



##############################################################
# TEST OPPORTUNITY SCORE RANGE
##############################################################

def test_opportunity_score_range(
    scored_df,
):
    """
    Opportunity score should be
    between 0 and 100.
    """

    scores = scored_df[

        "opportunity_score"

    ]


    assert scores.min() >= 0


    assert scores.max() <= 100



##############################################################
# TEST SCORE IS NUMERIC
##############################################################

def test_opportunity_score_numeric(
    scored_df,
):
    """
    Score must be numeric for
    sorting and dashboard filtering.
    """

    assert (

        pd.api.types

        .is_numeric_dtype(

            scored_df[
                "opportunity_score"
            ]

        )

    )



##############################################################
# TEST TIER ASSIGNMENT
##############################################################

def test_tier_assignment(
    scored_df,
):
    """
    Validate investment tiers.
    """

    valid_tiers = {

        "Low",

        "Fair",

        "Good",

        "Excellent",

    }


    actual_tiers = set(

        scored_df["tier"]

    )


    assert actual_tiers.issubset(

        valid_tiers

    )



##############################################################
# TEST HIGH SCORE GETS HIGH TIER
##############################################################

def test_score_tier_relationship():

    """
    Verify tier boundaries.

    70+
        Excellent

    50-70
        Good

    30-50
        Fair

    <30
        Low

    """

    test_df = pd.DataFrame(

        {

            "opportunity_score":[

                20,

                40,

                60,

                80,

            ]

        }

    )


    test_df["tier"] = pd.cut(

        test_df[
            "opportunity_score"
        ],

        bins=[

            0,

            30,

            50,

            70,

            101

        ],

        labels=[

            "Low",

            "Fair",

            "Good",

            "Excellent",

        ],

        include_lowest=True,

    ).astype(str)



    assert test_df.loc[0,"tier"] == "Low"


    assert test_df.loc[1,"tier"] == "Fair"


    assert test_df.loc[2,"tier"] == "Good"


    assert test_df.loc[3,"tier"] == "Excellent"



##############################################################
# TEST PERCENTILE GENERATION
##############################################################

def test_percentile_range(
    scored_df,
):
    """
    Percentile should be:

    0 <= percentile <= 100
    """

    percentile = scored_df[

        "percentile"

    ]


    assert percentile.min() >= 0


    assert percentile.max() <= 100



##############################################################
# TEST SORTING TOP INVESTMENT
##############################################################

def test_best_properties_rank_high(
    scored_df,
):
    """
    Highest opportunity score
    should appear first after sorting.
    """

    ranked = scored_df.sort_values(

        "opportunity_score",

        ascending=False,

    )


    assert (

        ranked.iloc[0]

        [

            "opportunity_score"

        ]

        >=

        ranked.iloc[-1]

        [

            "opportunity_score"

        ]

    )



##############################################################
# TEST NO MISSING SCORES
##############################################################

def test_no_missing_scores(
    scored_df,
):
    """
    Dashboard cannot work with
    missing ranking values.
    """

    score_columns = [

        "opportunity_score",

        "value_gap_score",

        "growth_score",

        "access_score",

        "safety_score",

    ]


    for column in score_columns:

        assert (

            scored_df[column]

            .isna()

            .sum()

            == 0

        )
        
        
