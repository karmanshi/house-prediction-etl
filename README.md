# 🏡 REIOS — Real Estate Investment Opportunity Scorer

> **An AI-powered Real Estate Investment Analysis System that identifies undervalued properties using Machine Learning, Anomaly Detection, and Composite Investment Scoring.**

---

# Overview

REIOS (Real Estate Investment Opportunity Scorer) is an end-to-end machine learning application that evaluates residential properties and ranks them according to their investment potential.

Instead of simply predicting property prices, REIOS combines multiple machine learning models with domain-specific scoring logic to identify properties that may represent strong investment opportunities.

The system is designed as a production-oriented pipeline consisting of:

* Data Cleaning (ETL)
* Feature Engineering
* Hedonic Price Prediction
* Anomaly Detection
* Composite Opportunity Scoring
* Investment Tier Classification
* FastAPI Backend
* Streamlit Dashboard

---

# Problem Statement

Traditional real estate platforms only display property listings and prices.

They do not answer questions such as:

* Which property is undervalued?
* Which area is growing rapidly?
* Which property has high investment potential?
* Which property offers the best value for money?

REIOS addresses these challenges using machine learning and statistical analysis.

---

# Objectives

The primary objectives of this project are:

* Predict fair market value of a property.
* Detect unusual property listings.
* Estimate investment opportunity.
* Rank properties from best to worst.
* Classify investment quality into tiers.
* Provide an interactive dashboard for investors.

---

# Project Architecture

```
Dataset
    │
    ▼
ETL Pipeline
    │
    ▼
Feature Engineering
    │
    ▼
─────────────────────────────────────
│        Machine Learning Models     │
│                                    │
│ 1. Hedonic Price Model             │
│ 2. Isolation Forest                │
│ 3. Opportunity Scoring             │
│ 4. Tier Classifier                 │
─────────────────────────────────────
    │
    ▼
FastAPI Backend
    │
    ▼
Streamlit Dashboard
```

---

# Machine Learning Pipeline

## Model 1 — Hedonic Price Model

Purpose:

Predict the expected market value of a property.

Algorithm:

* LightGBM Regressor

Output:

* Predicted Price
* Residual Percentage

Residual:

```
Residual % =
(Actual Price − Predicted Price)
-------------------------------- × 100
Predicted Price
```

Negative residuals indicate potentially undervalued properties.

---

## Model 2 — Anomaly Detection

Purpose:

Detect properties that significantly differ from the market.

Algorithm:

* Isolation Forest

Outputs:

* Anomaly Label
* Anomaly Score

Possible labels:

* 1 → Normal
* -1 → Anomalous

---

## Model 3 — Composite Opportunity Scoring

Several investment factors are normalized and combined into a single investment score.

Factors include:

* Value Gap Score
* Growth Score
* Accessibility Score
* Safety Score
* Anomaly Score

The weighted score is converted into an Opportunity Score between **0 and 100**.

---

## Model 4 — Investment Tier Classification

Properties are classified into investment categories.

Current tiers:

* Low
* Fair
* Good
* Excellent

Classification is performed using a LightGBM Classifier.

---

# Feature Engineering

The feature engineering pipeline generates features used by all machine learning models.

Examples include:

* Price per square foot
* Location encoding
* Property type encoding
* Rental yield
* Accessibility metrics
* Crime statistics
* Neighborhood growth indicators
* Amenity encoding
* Composite engineered features

---

# Project Structure

```
RealEstateInvestmentOpportunityScorer/

├── backend/
├── frontend/
├── training/
├── data/
├── models/
├── metrics/
├── evaluation_plots/
├── configs/
├── tests/
├── engineered_features/
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

# Technologies Used

### Programming

* Python

### Machine Learning

* LightGBM
* Isolation Forest
* Scikit-learn

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib

### API

* FastAPI

### Dashboard

* Streamlit

### Experiment Tracking

* MLflow

### Testing

* PyTest

### Containerization

* Docker

### Version Control

* Git
* GitHub

---

# Repository Workflow

```
Raw Dataset

      ↓

ETL

      ↓

Feature Engineering

      ↓

Train Models

      ↓

Save Models

      ↓

Composite Scoring

      ↓

FastAPI

      ↓

Streamlit Dashboard
```

---

# Testing

The project includes automated tests covering:

* Feature Engineering
* Model Loading
* Model Prediction
* Composite Scoring
* API Validation

Testing framework:

* PyTest

---

# Future Improvements

Potential future enhancements include:

* Deep Learning based price prediction
* Real-time property data ingestion
* Time-series forecasting
* Recommendation engine
* Interactive GIS mapping
* Cloud deployment
* Automated retraining pipeline

---

# License

This project is intended for educational and portfolio purposes.

Please review the repository license before commercial use.
