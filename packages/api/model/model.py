import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils import resample
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

df = pd.read_csv("data/adani.csv")

unique_symbols = {}

for sym in df["symbol"].unique():
    unique_symbols[sym] = df[df["symbol"] == sym]
    unique_symbols[sym].reset_index(inplace=True, drop=True)

updated_unique_stocks = {}

for sym, stock in unique_symbols.items():
    updated_unique_stocks[sym] = stock[
        ["open", "high", "low", "close", "volume", "dividends", "stock_splits"]
    ]

scaled_unique_stocks = {}

for sym, stock in updated_unique_stocks.items():
    scaler = MinMaxScaler()

    scaler.fit(stock)

    scaled_unique_stocks[sym] = scaler.transform(stock)

transformed_unique_stocks = {}

for sym, stock in scaled_unique_stocks.items():
    pca = PCA(n_components=2)

    pca.fit(stock)

    transformed_unique_stocks[sym] = pca.transform(stock)

df_pca = {}

for sym, stock in transformed_unique_stocks.items():
    df_pca[sym] = pd.DataFrame(stock, columns=["Feat_1", "Feat_2"])

"""# Building an IsolationForest Model for the above two Features"""

df_isolation_forest = copy.deepcopy(df_pca)

anomaly_inputs = ["Feat_1", "Feat_2"]

if_model = IsolationForest(contamination=0.005, random_state=42)

for sym, stock in df_isolation_forest.items():

    if_model.fit(stock[anomaly_inputs])

    stock["anomaly_score"] = if_model.decision_function(stock[anomaly_inputs])

    stock["anomaly"] = if_model.predict(stock[anomaly_inputs])


stock_with_fraud_isolation_forest = {}

for sym in df_isolation_forest:
    stock_with_fraud_isolation_forest[sym] = pd.merge(
        unique_symbols[sym][["timestamp", "symbol", "company"]],
        df_isolation_forest[sym],
        left_index=True,
        right_index=True,
    )
    stock_with_fraud_isolation_forest[sym]["timestamp"] = pd.to_datetime(
        stock_with_fraud_isolation_forest[sym]["timestamp"], unit="ns"
    )

for sym in stock_with_fraud_isolation_forest:
    print(
        stock_with_fraud_isolation_forest[sym][
            stock_with_fraud_isolation_forest[sym]["anomaly"] == -1
        ]
    )

stock_with_fraud_isolation_forest["ACC"][
    stock_with_fraud_isolation_forest["ACC"]["anomaly"] == -1
].head()

stock_with_fraud_isolation_forest["NDTV"][
    stock_with_fraud_isolation_forest["NDTV"]["anomaly"] == -1
].head()


df_svm = copy.deepcopy(df_pca)

svm_model = OneClassSVM(nu=0.005)

for sym, stock in df_svm.items():
    svm_model.fit(stock)

    stock["anomaly"] = svm_model.predict(stock)

stock_with_fraud_svm = {}

for sym in df_svm:
    stock_with_fraud_svm[sym] = pd.merge(
        unique_symbols[sym][["timestamp", "symbol", "company"]],
        df_svm[sym],
        left_index=True,
        right_index=True,
    )
    stock_with_fraud_svm[sym]["timestamp"] = pd.to_datetime(
        stock_with_fraud_svm[sym]["timestamp"], unit="ns"
    )

for sym in stock_with_fraud_svm:
    print(stock_with_fraud_svm[sym][stock_with_fraud_svm[sym]["anomaly"] == -1])

stock_with_fraud_svm["ACC"][stock_with_fraud_svm["ACC"]["anomaly"] == -1].head()

stock_with_fraud_svm["NDTV"][stock_with_fraud_svm["NDTV"]["anomaly"] == -1].head()

fraud_using_isolation_forest = 0
fraud_using_svm = 0

for sym, stock in stock_with_fraud_isolation_forest.items():
    fraud_using_isolation_forest += len(stock[stock["anomaly"] == -1])

for sym, stock in stock_with_fraud_svm.items():
    fraud_using_svm += len(stock[stock["anomaly"] == -1])

print(fraud_using_isolation_forest)
print(fraud_using_svm)

df_lof = copy.deepcopy(df_pca)

anomaly_inputs = ["Feat_1", "Feat_2"]

n_neighbors = 20

lof_model = LocalOutlierFactor(contamination=0.005, n_neighbors=n_neighbors)

for sym, stock in df_lof.items():
    stock["anomaly_lof"] = lof_model.fit_predict(stock[anomaly_inputs])

stock_with_fraud_lof = {}

for sym in df_lof:
    stock_with_fraud_lof[sym] = pd.merge(
        unique_symbols[sym][["timestamp", "symbol", "company"]],
        df_lof[sym],
        left_index=True,
        right_index=True,
    )
    stock_with_fraud_lof[sym]["timestamp"] = pd.to_datetime(
        stock_with_fraud_lof[sym]["timestamp"], unit="ns"
    )

for sym in stock_with_fraud_lof:
    print(stock_with_fraud_lof[sym][stock_with_fraud_lof[sym]["anomaly_lof"] == -1])

fraud_using_lof = 0

for sym, stock in stock_with_fraud_lof.items():
    fraud_using_lof += len(stock[stock["anomaly_lof"] == -1])

print(fraud_using_lof)

stock_with_fraud_lof["ACC"][stock_with_fraud_lof["ACC"]["anomaly_lof"] == -1].head()

stock_with_fraud_lof["NDTV"][stock_with_fraud_lof["NDTV"]["anomaly_lof"] == -1].head()

"""# Common Anomalies"""

combined_df = {}
common_anomalies_count = 0

for sym in df["symbol"].unique():
    combined_df[sym] = pd.merge(
        stock_with_fraud_isolation_forest[sym][
            stock_with_fraud_isolation_forest[sym]["anomaly"] == -1
        ],
        stock_with_fraud_svm[sym][stock_with_fraud_svm[sym]["anomaly"] == -1],
        on="timestamp",
    )

    combined_df[sym] = pd.merge(
        combined_df[sym],
        stock_with_fraud_lof[sym][stock_with_fraud_lof[sym]["anomaly_lof"] == -1],
        on="timestamp",
    )

    common_anomalies_count += len(combined_df[sym])

print("Number of common anomalies:", common_anomalies_count)

combined_df["ACC"].head()

combined_df["NDTV"].head()

df_ensemble = copy.deepcopy(df_pca)

anomaly_inputs = ["Feat_1", "Feat_2"]

num_bags = 3

isolation_forest_predictions = {}
svm_predictions = {}
lof_predictions = {}


for sym, stock in df_ensemble.items():

    isolation_forest_predictions[sym] = []
    svm_predictions[sym] = []
    lof_predictions[sym] = []

    for i in range(num_bags):
        sampled_data = resample(stock[anomaly_inputs], replace=True, random_state=i)

        isolation_forest_model = IsolationForest()
        isolation_forest_model.fit(sampled_data)

        isolation_forest_scores = isolation_forest_model.decision_function(
            stock[anomaly_inputs]
        )
        isolation_forest_predictions[sym].append(isolation_forest_scores)

        svm_model = OneClassSVM()
        svm_model.fit(sampled_data)

        svm_scores = svm_model.decision_function(stock[anomaly_inputs])
        svm_predictions[sym].append(svm_scores)

        lof_model = LocalOutlierFactor()
        lof_model.fit(sampled_data)

        lof_scores = lof_model.negative_outlier_factor_
        lof_predictions[sym].append(lof_scores)

isolation_forest_average = {}
svm_average = {}
lof_average = {}
ensemble_scores = {}
ensemble_predictions = {}

for sym, stock in df_ensemble.items():
    isolation_forest_average[sym] = sum(isolation_forest_predictions[sym]) / num_bags
    svm_average[sym] = sum(svm_predictions[sym]) / num_bags
    lof_average[sym] = sum(lof_predictions[sym]) / num_bags

    ensemble_scores[sym] = (
        isolation_forest_average[sym] + svm_average[sym] + lof_average[sym]
    ) / 3

    threshold = np.percentile(ensemble_scores[sym], 1)

    ensemble_predictions[sym] = np.where(ensemble_scores[sym] > threshold, 1, -1)

    stock["Ensemble_Anomaly"] = ensemble_predictions[sym]


stock_with_fraud_ensemble = {}

for sym in df_lof:
    stock_with_fraud_ensemble[sym] = pd.merge(
        unique_symbols[sym][["timestamp", "symbol", "company"]],
        df_ensemble[sym],
        left_index=True,
        right_index=True,
    )
    stock_with_fraud_ensemble[sym]["timestamp"] = pd.to_datetime(
        stock_with_fraud_ensemble[sym]["timestamp"], unit="ns"
    )

ensemble_fraud_count = 0
total_data_points = 0

for sym, stock in stock_with_fraud_ensemble.items():
    total_data_points += len(stock)
    ensemble_fraud_count += len(stock[stock["Ensemble_Anomaly"] == -1])

print(ensemble_fraud_count)
print(total_data_points)

stock_with_fraud_ensemble["ACC"][
    stock_with_fraud_ensemble["ACC"]["Ensemble_Anomaly"] == -1
].head()

stock_with_fraud_ensemble["NDTV"][
    stock_with_fraud_ensemble["NDTV"]["Ensemble_Anomaly"] == -1
].head()

silhouette_scores = []
db_scores = []
ch_scores = []

for sym, stock in df_ensemble.items():

    silhouette = silhouette_score(stock[anomaly_inputs], stock["Ensemble_Anomaly"])
    silhouette_scores.append(silhouette)

    db_index = davies_bouldin_score(stock[anomaly_inputs], stock["Ensemble_Anomaly"])
    db_scores.append(db_index)

    ch_index = calinski_harabasz_score(stock[anomaly_inputs], stock["Ensemble_Anomaly"])
    ch_scores.append(ch_index)

mean_silhouette = np.mean(silhouette_scores)
mean_db = np.mean(db_scores)
mean_ch = np.mean(ch_scores)

print("Mean Silhouette Score:", mean_silhouette)
print("Mean Davies-Bouldin Index:", mean_db)
print("Mean Calinski-Harabasz Index:", mean_ch)
