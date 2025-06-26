# FX Forecasting Pipeline

An end-to-end automated data pipeline to forecast foreign exchange (FX) currency pair movements using regression and classification models. This project leverages real-time market data, macroeconomic indicators, and lightweight AutoML to deliver directional predictions.

## 🔍 Overview

This project focuses on predicting FX currency pair movements using:
- Regression models for base pairs (e.g., EUR/USD)
- Classification models for synthetic pairs
- Macroeconomic feature engineering, including BTC correlation

The pipeline is orchestrated using **Apache Airflow**, stores time-series data with **ArcticDB**, and uses **PyCaret** for fast, modular model training.

---

## ⚙️ Tools & Technologies

- **Python** – Data handling, modeling, automation
- **PyCaret** – Automated regression and classification
- **Polygon API** – Real-time FX and BTC market data
- **ArcticDB** – High-performance time-series data storage
- **SQLite** – Local relational storage for structured data
- **Apache Airflow** – DAG scheduling for model runs
- **Pandas, NumPy, Seaborn** – Data manipulation & visualization

---

## 🧠 Features Engineered

- Keltner Channel indicators
- Rolling window statistics (mean, volatility, momentum)
- BTC/USD correlation as macro proxy
- Lag features and price ratios
- Labeling rule for directional classification

---

## 📈 Model Performance

| Model Type     | Target           | Metric         | Score      |
|----------------|------------------|----------------|------------|
| Regression     | Base pairs       | R² / MAE       | ~0.68 / ~0.02 |
| Classification | Synthetic pairs  | F1 Score       | ~0.82     |
| Overall        | Directional acc. | Accuracy       | ~78%      |

*Tested on holdout set over a 30-day period.*

---
