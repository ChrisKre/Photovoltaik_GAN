"""
Durchführung der manuellen Anonymisierung
 - Dazu werden die scaler objekte der jeweiligen Standote benötigt. Daher zunächst src/anonymize/manuelle_ano/create_scaler.py ausführen
"""

import os.path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.anonymize.manuelle_ano.utils import interpolate_to_koethen
from src.data.settings import processed_dir, ano_dir
from src.utils import check_make_dir

if __name__ == "__main__":
    locations = ["Le_Havre", "Madrid", "München"]
    norm_location = "Koethen"

    # Load Scaler
    norm_scaler = joblib.load(os.path.join("scaler", f"{norm_location}_scaler.pkl"))
    # Load data of normlocation
    fname_location = os.path.join(processed_dir, f"{norm_location}.csv")
    df_norm_location = pd.read_csv(fname_location, index_col=0)

    # For all locations run anonymization
    for location in locations:
        # Load location data
        fname_location = os.path.join(processed_dir, f"{location}.csv")
        df_location = pd.read_csv(fname_location, index_col=0)

        # Scale data
        scaler = MinMaxScaler()
        location_values = scaler.fit_transform(
            np.expand_dims(df_location.values.flatten(), 1)
        )

        # Rescale with normlocation scaler
        location_values_rescaled = norm_scaler.inverse_transform(location_values)
        # Group data by day
        daily_location_values_rescaled = np.reshape(
            location_values_rescaled, (4383, 24)
        )

        normed_data = []
        # Stretch the daily sequences
        for i, (day, day_norm) in enumerate(
            zip(daily_location_values_rescaled, df_norm_location.values)
        ):
            try:
                daily_location_values_normed = interpolate_to_koethen(
                    np.expand_dims(day, 1), np.expand_dims(day_norm, 1)
                )
                normed_data.append(daily_location_values_normed)
                print(f"{location}-{df_norm_location.index[i]}:")
            except:
                print(f"Fehler für Standort: {location} und Stichprobe: {i}")
                normed_data.append(np.expand_dims(day, 1))
        # Save the anonymized data
        df_location_normed = pd.DataFrame(
            np.squeeze(normed_data), index=df_location.index
        )
        fname_location_normed = os.path.join(ano_dir, "manuelle", f"{location}.csv")
        check_make_dir(fname_location_normed, True)
        df_location_normed.to_csv(fname_location_normed, index=True)
