import os.path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.anonymize.manuelle_ano.utils import interpolate_to_koethen
from src.data.settings import processed_dir



if __name__ == "__main__":
    locations = ["München"]
    norm_location = "Koethen"
    norm_scaler = joblib.load(os.path.join('..', "scaler", f"{norm_location}_scaler.pkl"))

    fname_location = os.path.join(processed_dir, f"{norm_location}.csv")
    df_norm_location = pd.read_csv(fname_location, index_col=0)
    df_norm_location.index = pd.to_datetime(df_norm_location.index)

    # dates_le_havre = [pd.Timestamp(year=2016, month=1, day=2), pd.Timestamp(year=2016, month=2, day=12),
    #                  pd.Timestamp(year=2016, month=4, day=23), pd.Timestamp(year=2016, month=5, day=9),
    #                  pd.Timestamp(year=2016, month=6, day=18), pd.Timestamp(year=2016, month=7, day=13),
    #                  pd.Timestamp(year=2016, month=9, day=29), pd.Timestamp(year=2016, month=10, day=30)]

    # dates = [pd.Timestamp(year=2016, month=1, day=29), pd.Timestamp(year=2016, month=2, day=13),
    #                 pd.Timestamp(year=2016, month=4, day=4), pd.Timestamp(year=2016, month=5, day=4),
    #                 pd.Timestamp(year=2016, month=6, day=30), pd.Timestamp(year=2016, month=7, day=7),
    #                 pd.Timestamp(year=2016, month=9, day=11), pd.Timestamp(year=2016, month=10, day=18)]

    dates = [
        pd.Timestamp(year=2016, month=1, day=23),
        pd.Timestamp(year=2016, month=12, day=27),
        pd.Timestamp(year=2016, month=3, day=12),
        pd.Timestamp(year=2016, month=4, day=27),
        pd.Timestamp(year=2016, month=6, day=28),
        pd.Timestamp(year=2016, month=7, day=7),
        pd.Timestamp(year=2016, month=9, day=14),
        pd.Timestamp(year=2016, month=11, day=20),
    ]

    plot_index = [i for i in range(0, 24)]

    for location in locations:
        fname_location = os.path.join(processed_dir, f"{location}.csv")
        df_location = pd.read_csv(fname_location, index_col=0)
        df_location.index = pd.to_datetime(df_location.index)

        location_scaler = joblib.load(os.path.join('..', "scaler", f"{location}_scaler.pkl"))

        fig, ax = plt.subplots(nrows=4, ncols=len(dates))

        location_txt = ""
        for i, date in enumerate(dates):
            normday = df_norm_location[df_norm_location.index == date]

            day = df_location[df_location.index == date]
            location_values = day.values.flatten()
            location_values_scaled = location_scaler.transform(
                np.expand_dims(location_values, 1)
            )
            location_values_rescaled = norm_scaler.inverse_transform(
                location_values_scaled
            )
            daily_location_values_normed = interpolate_to_koethen(
                location_values_rescaled, normday.values[0]
            )

            ax[0, i].plot(plot_index, location_values)
            ax[1, i].plot(plot_index, location_values_scaled)
            ax[2, i].plot(plot_index, location_values_rescaled)
            ax[3, i].plot(plot_index, daily_location_values_normed)


        plt.show()
