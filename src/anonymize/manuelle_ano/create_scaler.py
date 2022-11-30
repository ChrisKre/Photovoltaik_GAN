"""
Erstelle die benötigten Scaler Objekte für die Manuelle Skalierung
"""

import os.path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.data.settings import processed_dir
from src.utils import check_make_dir

if __name__ == "__main__":
    locations = ["Koethen", "Le_Havre", "Madrid", "München"]
    # Iterate over all locations
    for location in locations:
        # Load location data
        fname_location = os.path.join(processed_dir, f"{location}.csv")
        df_location = pd.read_csv(fname_location, index_col=0)
        location_values = df_location.values.flatten()
        # Scale data
        scaler = MinMaxScaler()
        scaled_location_values = scaler.fit_transform(
            np.expand_dims(location_values, 1)
        )
        scaler_fname = os.path.join("scaler", f"{location}_scaler.pkl")
        check_make_dir(scaler_fname, True)
        # Save scaler for later use
        joblib.dump(scaler, scaler_fname)
