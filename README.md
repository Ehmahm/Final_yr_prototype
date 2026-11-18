# Ensemble Knowledge Distillation and Cloud-Native Deployment for Peak Load Demand Forecasting

**Author:** Emmanuel Magomu  
**Reg. No.:** M24B38/011 | **Student No.:** B27294  
**Institution:** Uganda Christian University — Faculty of Engineering, Design and Technology  
**Programme:** BSc. Data Science and Analytics (AI & Machine Learning track)  
**Advisor:** Dr. Innocent Ndibatya

---

## Overview

Peak electrical load demand forecasting is a core challenge in smart grid management. State-of-the-art ensemble models achieve high accuracy but are too large to run on resource-constrained edge devices such as IoT gateways or substation hardware.

This project builds an end-to-end pipeline that:

1. Learns temporal features from smart meter and grid load data
2. Trains a heterogeneous teacher ensemble (XGBoost + LightGBM + BiLSTM)
3. Distills the ensemble into a compact student MLP via structured knowledge distillation
4. Exports the student to ONNX for edge inference
5. Serves the model through a FastAPI microservice with a CSV replay client that simulates synchronous IoT input
6. Deploys via Docker and a local AWS emulator (Floci) for S3, ECS, and CloudWatch

The long-term goal is a practical, edge-deployable forecasting tool for grid operators in Uganda and sub-Saharan Africa.

---

## Datasets

| Dataset | Level | Records | Target | Path |
|---|---|---|---|---|
| UCI Household Electric Power Consumption | Household | 2.07M (minute) → 34k (hourly) | `Global_active_power` (kW) | `data/raw/household_power_consumption.txt` |
| ELIA Open Data (`ods001`) | National grid (Belgium) | 190k (15-min) → 48k (hourly) | `Total Load` (MW) | `data/raw/elia_raw_export.csv` |

Both datasets are public and licensed for research use.

---

## Pipeline
