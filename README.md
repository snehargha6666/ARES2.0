# ARES 2.0

## AI-Based Disaster Response and Severity Prediction System

ARES 2.0 is an Artificial Intelligence and Machine Learning based disaster analysis system designed to assist in understanding disaster severity and supporting data-driven disaster response.

The project combines machine learning, data analysis, and a web-based interface to process disaster-related information and generate severity predictions.

---

## Project Objectives

The main objectives of ARES 2.0 are:

- Predict the severity of disasters using historical disaster data.
- Perform data preprocessing and exploratory data analysis.
- Identify important disaster-related features.
- Train and evaluate machine learning models.
- Provide disaster severity predictions through an accessible interface.
- Build a foundation for future image-based and time-series disaster analysis.

---

## Current Implementation

The current version focuses primarily on:

### Module 1 — Disaster Severity Prediction

Technologies and concepts currently implemented include:

- Exploratory Data Analysis
- Feature Engineering
- Principal Component Analysis (PCA)
- Random Forest
- Artificial Neural Network (ANN)
- Multi-Layer Perceptron (MLP)
- Model evaluation
- Disaster severity scoring
- Feature importance analysis
- Prediction API
- Web-based frontend

---

## System Architecture

```text
                 Disaster Data
                      |
                      v
              Data Preprocessing
                      |
                      v
             Feature Engineering
                      |
             +--------+--------+
             |                 |
             v                 v
        Random Forest        MLP / ANN
             |                 |
             +--------+--------+
                      |
                      v
             Severity Prediction
                      |
                      v
                 Backend API
                      |
                      v
                  Frontend
                      |
                      v
               User Interface
