import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dagster import asset, Definitions, define_asset_job, AssetExecutionContext
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os

@asset
def load_data():
    df = pd.read_csv("/content/taxi_trips_2024.csv")
    df.columns = [c.replace(" ", "_").lower() for c in df.columns]
    df = df.dropna(subset=['trip_seconds', 'trip_miles', 'fare', 'trip_total'])
    return df

@asset
def preprocess_data(load_data):
    df = load_data
    features = ['trip_seconds', 'trip_miles', 'tips', 'tolls', 'extras']
    target = 'trip_total'

    X = df[features]
    y = df[target]

    return train_test_split(X, y, test_size=0.2, random_state=42)

@asset
def generate_eda(load_data):
    df = load_data

    df_clean = df[(df['trip_total'] > 0) & (df['trip_total'] < 150)]
    df_clean = df_clean[(df['trip_miles'] > 0) & (df['trip_miles'] < 50)]
    
    if not os.path.exists("eda_plots"):
        os.makedirs("eda_plots")
    
    #Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df_clean[['trip_seconds', 'trip_miles', 'fare', 'tips', 'trip_total']]
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig("eda_plots/correlation_heatmap.png")
    plt.close()
    
    #Distribution of Trip Total
    plt.figure(figsize=(10, 6))
    sns.histplot(df_clean['trip_total'], bins=50, kde=True, color='teal')
    plt.title("Distribution of Trip Fares (Filtered < $150)")
    plt.xlabel("Total Fare ($)")
    plt.savefig("eda_plots/trip_total_dist.png")
    plt.close()

    return "EDA plots saved to /eda_plots folder"

@asset
def train_decision_tree(preprocess_data):
    X_train, X_test, y_train, y_test = preprocess_data
    model = DecisionTreeRegressor(max_depth=5)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = mean_absolute_error(y_test, preds)
    return {"model": model, "mae": score, "name": "Decision Tree"}

@asset
def train_random_forest(preprocess_data):
    X_train, X_test, y_train, y_test = preprocess_data
    model = RandomForestRegressor(n_estimators=50, max_depth=10)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = mean_absolute_error(y_test, preds)
    return {"model": model, "mae": score, "name": "Random Forest"}

@asset
def train_linear_regression(preprocess_data):
    X_train, X_test, y_train, y_test = preprocess_data
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = mean_absolute_error(y_test, preds)
    return {"model": model, "mae": score, "name": "Linear Regression"}

@asset
def compare_models(train_decision_tree, train_random_forest, train_linear_regression):
    results = [train_decision_tree, train_random_forest, train_linear_regression]

    best_model = min(results, key=lambda x: x['mae'])

    report = f"Model Comparison Report:\n"
    for res in results:
        report += f"{res['name']}: MAE = {res['mae']:.4f}\n"

    report += f"\nWINNER: {best_model['name']} with MAE {best_model['mae']:.4f}"
    with open("model_report.txt", "w") as f:
        f.write(report)

    print(report)
    return report

defs = Definitions(
    assets=[
        load_data,
        preprocess_data,
        generate_eda,
        train_decision_tree,
        train_random_forest,
        train_linear_regression,
        compare_models
    ]

)
