# Dagster ML Pipeline: Chicago Taxi Trip Estimation

## Overview
This project implements a reproducible Machine Learning pipeline using **Dagster** to predict taxi fares based on the [Chicago Taxi Trips 2024 dataset](https://www.kaggle.com/datasets/adelanseur/taxi-trips-chicago-2024).

Unlike fragile Jupyter Notebooks, this pipeline treats data and models as **Ops & Job**, allowing for:
- **Partial Re-runs:** Tweak a model without reloading data.
- **Data Lineage:** Automatic tracking of data flow.
- **Reproducibility:** Code configuration is versioned.

## Pipeline Architecture
The pipeline consists of the following assets:
1. `load_data`: Ingests raw CSV data.
2. `preprocess_data`: Cleans null values and performs train-test splits.
3. `train_decision_tree, train_random_forest, train_linear_regression`: Trains Decision Tree, Random Forest, and Linear Regression models in parallel.
4. `compare_models`: Evaluates MAE and selects the best model.

## Results & Efficiency
Moving to Dagster reduced iteration time by skipping redundant data processing steps.

| Run Type | Duration | Description |
| :--- | :--- | :--- |
| **Full Run** | 1 min 38s | Loads data, cleans, trains all models. |
| **Partial Run** | **29s** | Skips loading/cleaning. Only retrains models. |
| **Efficiency** | **~70%** | Time saved per iteration. |

##  How to Run
1. Install dependencies: `pip install dagster pandas scikit-learn seaborn`
2. Run the pipeline UI: `dagster dev -f pipeline.py`
3. Click **"Launch Run"** to execute.
