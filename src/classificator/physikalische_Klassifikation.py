"""
Standortvorhersage mit physikalischen Gesetzen
"""

import os
from math import sin, cos, sqrt, atan2, radians

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pvsystemprofiler.estimator import ConfigurationEstimator
from solardatatools import DataHandler

from src.data.settings import merged_dir, location_long_lat

köthen_true = (
    location_long_lat["Koethen"]["lat"],
    location_long_lat["Koethen"]["long"],
)
le_havre_true = (
    location_long_lat["Le_Havre"]["lat"],
    location_long_lat["Le_Havre"]["long"],
)
madrid_true = (location_long_lat["Madrid"]["lat"], location_long_lat["Madrid"]["long"])
münchen_true = (
    location_long_lat["München"]["lat"],
    location_long_lat["München"]["long"],
)


def distance(pred, true):
    R = 6373.0

    lat1 = radians(pred[0])
    lon1 = radians(pred[1])
    lat2 = radians(true[0])
    lon2 = radians(true[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance


def get_lon_lat(df: pd.DataFrame):
    dh_ac_power = DataHandler(df)
    dh_ac_power.run_pipeline(power_col=("GlobInc"), solver="MOSEK", verbose=True)
    dh_ac_power.report()
    # dh_ac_power.plot_heatmap(matrix='filled', units="W/m²", figsize=(10, 5))

    est_power = ConfigurationEstimator(dh_ac_power, gmt_offset=+2, data_matrix="raw")
    est_power.estimate_longitude(estimator="calculated")
    lon = est_power.longitude

    # dh_ac_power.plot_circ_dist(flag='clear')
    plt.show()
    est_power.estimate_latitude()
    lat = est_power.latitude
    return lon, lat


def location_predict(df):
    predictions = []
    lon_predictions, lat_predictions = get_lon_lat(df)
    for lon, lat in zip(lon_predictions, lat_predictions):
        köthen = distance((lon, lat), köthen_true)
        le_havre = distance((lon, lat), le_havre_true)
        madrid = distance((lon, lat), madrid_true)
        münchen = distance((lon, lat), münchen_true)

        prediction = sorted([köthen, le_havre, madrid, münchen])[0]
        if prediction == köthen:
            predictions.append("Köthen")
        elif prediction == le_havre:
            predictions.append("Köthen")
        elif prediction == madrid:
            predictions.append("Madrid")
        else:
            predictions.append("München")
    return predictions


def predict_for_day(df):
    location_predict(df)


def to_add(df, column):
    to_add = []
    for v in df[column].unique():
        to_add1 = len(df[df[column] == v]) / len(df[column]) * (4383 - len(df[column]))
        to_add.append(int(to_add1))

    l = list(df[column])
    for v, ta in zip(df[column].unique(), to_add):
        l += list(np.full((ta), v))
    return l


if __name__ == "__main__":
    koethen_fname = os.path.join(merged_dir, "Koethen.csv")
    le_havre_fname = os.path.join(merged_dir, "Le_Havre.csv")
    madrid_fname = os.path.join(merged_dir, "Madrid.csv")
    muenchen_fname = os.path.join(merged_dir, "München.csv")

    results = pd.DataFrame()
    # df_le_havre = pd.read_csv(le_havre_fname, index_col=0)
    # df_le_havre.index = pd.to_datetime(df_le_havre.index)
    # result = location_predict(df_le_havre)
    # lh = pd.DataFrame({'Le_Havre': result})
    # results['Le_Havre'] = to_add(lh, 'Le_Havre')

    df_koethen = pd.read_csv(koethen_fname, index_col=0)
    df_koethen.index = pd.to_datetime(df_koethen.index)
    result = location_predict(df_koethen)
    lh = pd.DataFrame({"Köthen": result})
    results_k = to_add(lh, "Köthen")
    results["Köthen"] = results_k[0 : len(results["Le_Havre"])]

    df_madrid = pd.read_csv(madrid_fname, index_col=0)
    df_madrid.index = pd.to_datetime(df_madrid.index)
    result = location_predict(df_koethen)
    lh = pd.DataFrame({"Madrid": result})
    results_m = to_add(lh, "Madrid")
    results["Madrid"] = results_m[0 : len(results["Le_Havre"])]
    # results['Madrid'] = location_predict(df_madrid)

    df_muenchen = pd.read_csv(muenchen_fname, index_col=0)
    df_muenchen.index = pd.to_datetime(df_muenchen.index)
    result = location_predict(df_koethen)
    lh = pd.DataFrame({"München": result})
    results_mu = to_add(lh, "München")
    results["München"] = results_mu[0 : len(results["Le_Havre"])]

    results.to_csv("location_pred.csv")
