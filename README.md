# Squareshift's Chicago Taxi Services

## Overview
This project develops an AI solution for Chicago Taxi Services to predict tip amounts for taxi rides. It utilizes a large public dataset (184 Million Rows) located in Google Cloud BigQuery's Public Data.

## Problem Statement
In the urban transportation industry, particularly taxi services, tips are impactful metrics for both drivers and service providers. This project aims to predict tip amounts based on various trip features, providing valuable insights for the taxi industry.

## Solution
We developed a Neural Network using TensorFlow to predict tip amounts based on features like trip miles, fare, trip duration, and pickup/dropoff locations.

## Process Workflow
1. Project Set Up
2. Data Cleaning and Transformation
3. Data Visualization
4. Preparing Data for Model Training
5. TensorFlow Models
6. Evaluating for the Best Model

## Setup
### Requirements
- Use a separate project in GCP
- Data should reside in BigQuery (make a copy into liscensed GCP project)
- Data Exploration reports - visualization
- Identify Machine Learning use case
- Begin feature extraction and storage in Vertex AI Feature Store and Pre-processing in Dataflow or Dataproc (To scale the pre-processing from a small data sample to the whole dataset)
- Try different ML models, evaluate them and choose the best for production. Use Vertex AI for sampling, model training, model evaluation, hyperparameter tuning, serving. Document bias-variance trade-off decisions
- Code should be checked in to Github, use Google Cloud Certificate Signing Request (CSR), and include a ReadMe file. (push it to Github)
- Well documented - use cases, architecture, flow diagram, pre-processing pipeline, visualizations, results etc

### Installation
1. Create a new project in Google Cloud Console
2. Copy the Chicago Taxi Trips dataset into your project from BigQuery's public data
3. Set up Google Colab and authenticate with your Google Cloud account

## Data Preparation
- Data cleaning and transformation using Python
- Visualization with Pandas and Matplotlib
- Feature selection and engineering
- Principal Component Analysis (PCA)

## Models Tested
1. Support Vector Regression (SVR)
2. Keras Sequential Neural Network
3. Gradient Boosting Model
4. Polynomial Regression

## Results
SVR emerged as the best model for scaling to the full 184 million row dataset, balancing accuracy (R-squared: 0.7413) with scalability and efficiency in Google Cloud's Dataflow architecture.

## Usage
Run code in a Google Colab Notebook
location: us-central1

## Authors
- Sama Shah

## Acknowledgments
- SquareShift Technologies Inc
- Google Cloud Platform

## Resources
1. Project Summary Document: https://docs.google.com/document/d/1kqJwnKXWDRxIK152KfYMCf03fqy0JJNyk2hkW3rn3s4/edit
2. Live Documentation: https://docs.google.com/document/d/1dYRp1580HKj-3R-Sr2SmeZsdugDRXbNQVrFNHMiT-Vk/edit
3. Chicago Taxi Trips Demo Presentation Slides: https://docs.google.com/presentation/d/1EIAYosi-YihLX3T5MwiyOPqPB4XCecMYsKWouDleeC8/edit#slide=id.p

For more detailed information, please refer to the full project documentation (1).