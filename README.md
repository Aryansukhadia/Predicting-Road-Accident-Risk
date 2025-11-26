# Accident Risk Prediction Using Apache Spark MLlib  
### Gradient Boosted Trees (GBT) — Big Data Analytics Project  
**By:** Shashwat Gohel | Aryan Sukhadia | Vedant Parikh  
**Course:** CSE-521 — Big Data Analytics  

---

## Overview  
This repository contains our complete pipeline for **predicting accident risk on road segments** using **Apache Spark MLlib**, specifically the **Gradient Boosted Trees (GBT) Regressor**.

The goal is to assign each road segment a continuous **accident_risk score (0–1)** using features such as road curvature, speed limits, weather, lighting, time-of-day, and other environmental factors.  
This helps in identifying high-risk road segments and supporting road-safety planning.

---

---

## Dataset Summary  
- **Total segments:** ~690,000  
- **Training data:** 517,000 rows  
- **Test data:** 172,000 rows  
- **Total features:** 12  
  - 4 categorical  
  - 4 boolean  
  - 3 numeric  
  - 1 target: `accident_risk (0–1)`  

###  Most Influential Features  
- **Curvature** — strongest correlation with accident risk *(0.545)*  
- **Speed Limit** — significant positive correlation *(0.432)*

Urban roads show the highest median risk; highways have the lowest.

---

## Model Used — Gradient Boosted Trees (GBT)

### Why GBT?
- Learns complex non-linear patterns  
- Excellent performance on tabular data  
- Handles mixed feature types  
- Sequential boosting reduces error iteratively  
- Scales efficiently with **Spark MLlib**

### Key Hyperparameters  
- `maxIter` — number of boosting stages  
- `maxDepth` — depth of each tree  
- `stepSize` — learning rate  

---

## Pipeline Overview (from notebook)
1. Load and clean data  
2. Encode categorical variables  
3. Assemble feature vector using `VectorAssembler`  
4. Split into train/test  
5. Train GBTRegressor  
6. Evaluate model (RMSE, MAE, R²)  
7. Generate visualizations  
8. Save outputs  

---

## Final Results

| Metric | Score |
|--------|--------|
| **RMSE** | **0.06297** |
| **R²** | **0.8567** |
| **MAE** | **0.04865** |
| **Improvement over baseline** | **62.1%** |

The model demonstrates strong predictive capability on synthetic road-safety data.

---

##  Visual Insights  
The notebook and PPT include visualizations of:

- Accident Risk Distribution  
- Risk by Road Type  
- Risk vs Curvature  
- Risk vs Speed Limit  
- Lighting Conditions (Day/Night/Dim)  
- Weather Conditions (Clear/Rain/Fog)  
- Holiday vs Regular Days  
- Time-of-Day Risk Patterns  

These plots reinforce the impact of curvature, lighting, and speed on road accidents.

---

## Running the Notebook

### Requirements
- Python 3.x  
- Apache Spark 3.x  
- PySpark  
- Jupyter Notebook  

### Install PySpark
```bash
pip install pyspark

