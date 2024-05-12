import numpy as np
import pandas as pd
import copy
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils import resample

df = pd.read_csv("/home/josu/Desktop/Projects/StockGuardian/packages/api/model/data/adani.csv")

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

"""
    Building IsolationForest model

    With Feat_1 and Feat_2 from PCA

    Parameters:
        contamination: 0.005
        random_state: 42

    Result: 
        `stock_with_fraud_isolation_forest` contains following keys
        ["timestamp", "company", "symbol", "Feat_1", "Feat_2", "anomaly", "anomaly_score"]

        if `anomaly` is -1 then marked Fraud
"""

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

"""
    Building OneClassSVM model

    With Feat_1 and Feat_2 from PCA

    Parameters:
        nu: 0.005

    Result: 
        `stock_with_fraud_svm` contains following keys
        ["timestamp", "company", "symbol", "Feat_1", "Feat_2", "anomaly"]

        if `anomaly` is -1 then marked Fraud
"""

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

"""
    Building LocalOutlierFactor model

    With Feat_1 and Feat_2 from PCA

    Parameters:
        contamination: 0.005
        n_neighbors: 20

    Result: 
        `stock_with_fraud_lof` contains following keys
        ["timestamp", "company", "symbol", "Feat_1", "Feat_2", "anomaly_lof"]

        if `anomaly_lof` is -1 then marked Fraud
"""

df_lof = copy.deepcopy(df_pca)

anomaly_inputs = ["Feat_1", "Feat_2"]

lof_model = LocalOutlierFactor(contamination=0.005, n_neighbors=20)

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

"""
    Building Ensembled model

    With IsolationForest, OneClassSVM and LocalOutlierFactor model

    Parameters:
        num_bags: 3

    Result: 
        `stock_with_fraud_ensemble` contains following keys
        ["timestamp", "company", "symbol", "Feat_1", "Feat_2", "Ensemble_Anomaly"]

        if `Ensemble_Anomaly` is -1 then marked Fraud
"""

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
        unique_symbols[sym][["timestamp", "symbol", "company", "close"]],
        df_ensemble[sym],
        left_index=True,
        right_index=True,
    )
    # stock_with_fraud_ensemble[sym]["timestamp"] = pd.to_datetime(
    #     stock_with_fraud_ensemble[sym]["timestamp"], unit="ns"
    # )
    stock_with_fraud_ensemble[sym] = stock_with_fraud_ensemble[sym].to_dict('records')

with open('output.json', 'w') as fp:
    json.dump(stock_with_fraud_ensemble, fp, indent=2)

print("complete")
