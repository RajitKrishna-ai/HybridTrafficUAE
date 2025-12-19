# HybridTrafficUAE

An explainable machine learning system for urban traffic congestion prediction and cause attribution in UAE smart cities using hybrid real and synthetic data.

---

## Project Overview

Traffic congestion is a major challenge in rapidly growing smart cities.  
This project implements an **end-to-end ML system** to:

- Predict congestion levels (**Low / Medium / High**)  
- Attribute underlying causes (**Peak Hour, Weather, Incident, Holiday/Event**)  

It uses a **hybrid data strategy**, combining:

- 🟢 Real public traffic & weather data  
- 🟡 Synthetic/proxy data for incidents where real data is unavailable  

The system emphasizes **explainability, reproducibility, and production readiness**.

---

## Features

- Predicts traffic congestion levels and assigns causes  
- Uses a hybrid real + synthetic data strategy  
- Fully explainable via **SHAP**  
- Production-style pipeline with deterministic outputs  
- Minimal unit tests for data integrity and inference checks  

---


## Repository Structure

HybridTrafficUAE/
├── data/ processed data
├── models/ # saved models & feature configs
├── notebooks/ # EDA & model analysis
│ ├── 01_data_exploration.ipynb
│ └── 02_model_analysis.ipynb
├── outputs/ # final predictions & SHAPplots
├── src/ # python scripts: ingestion, features, preprocessing, pipeline
├── tests/ # minimal unit tests
├── environment.yml # conda environment
├── requirements.txt # pip dependencies
├── README.md
└── .gitignore


