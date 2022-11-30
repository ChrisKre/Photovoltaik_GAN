"""
- Beinhaltet Informationen die zum Laden der raw .csv benötigt werden
- Pathhandling
- Informationen über Geografische Position der vier Standorte
"""
import os
import pathlib

import numpy as np

project_dir = os.path.dirname(
    os.path.dirname(os.path.join(pathlib.Path(__file__).parent.resolve()))
)
data_dir = os.path.join(project_dir, "data")
raw_dir = os.path.join(data_dir, "raw")
merged_dir = os.path.join(data_dir, "merged")
processed_dir = os.path.join(data_dir, "processed")
synth_dir = os.path.join(data_dir, "synth")
ano_dir = os.path.join(data_dir, "ano")

pv_data_names = [
    "Datum",
    "GlobInc",
    "T_Amb",
    "WindVel",
    "AzSol",
    "AngInc",
    "HSol",
    "AngProf",
    "UArray",
    "IArray",
    "TArray",
    "E_Grid",
    "EOutInv",
    "EArray",
]
pv_data_dtype = {
    "Datum": str,
    "GlobInc": np.float,
    "T_Amb": np.float,
    "WindVel": np.float,
    "AzSol": np.float,
    "AngInc": np.float,
    "HSol": np.float,
    "AngProf": np.float,
    "UArray": np.float,
    "IArray": np.float,
    "TArray": np.float,
    "E_Grid": np.float,
    "EOutInv": np.float,
    "EArray": np.float,
}

pv_data_madrid_names = [
    "Datum",
    "GlobInc",
    "EArray",
    "E_Grid",
    "HSol",
    "AzSol",
    "AngInc",
    "AngProf",
    "T_Amb",
    "WindVel",
    "TExtON",
    "TArray",
    "IArray",
    "UArray",
    "EOutInv",
]
pv_data_madrid_dtype = {
    "Datum": str,
    "GlobInc": np.float,
    "EArray": np.float,
    "E_Grid": np.float,
    "HSol": np.float,
    "AzSol": np.float,
    "AngInc": np.float,
    "AngProf": np.float,
    "T_Amb": np.float,
    "WindVel": np.float,
    "TExtON": np.float,
    "TArray": np.float,
    "IArray": np.float,
    "UArray": np.float,
    "EOutInv": np.float,
}
locations = {
    "Madrid": {
        "dtype": pv_data_madrid_dtype,
        "names": pv_data_madrid_names,
        "decimal": ".",
    },
    "Koethen": {"dtype": pv_data_dtype, "names": pv_data_names, "decimal": "."},
    "München": {"dtype": pv_data_dtype, "names": pv_data_names, "decimal": ","},
    "Le_Havre": {"dtype": pv_data_dtype, "names": pv_data_names, "decimal": ","},
    # 'Berlin': {'dtype': pv_data_dtype, 'names': pv_data_names, 'decimal': '.'},
    # 'Halle': {'dtype': pv_data_dtype, 'names': pv_data_names, 'decimal': '.'},
    # 'Hamburg': {'dtype': pv_data_dtype, 'names': pv_data_names, 'decimal': ','},
}

classification_labels = {
    "Madrid": 0,
    "Koethen": 1,
    "München": 2,
    "Le_Havre": 3,
    # 'Berlin': 2,
    # 'Halle': 3,
    # 'Hamburg': 4,
}

location_long_lat = {
    "Madrid": {"long": -3.703790, "lat": 40.416775},
    "Koethen": {"long": 11.97093, "lat": 51.75185},
    "München": {"long": 11.576124, "lat": 48.137154},
    "Le_Havre": {"long": 0.1250, "lat": 49.4960},
    "Berlin": {"long": 13.404954, "lat": 52.520008},
    "Halle": {"long": 11.966667, "lat": 51.483334},
    "Hamburg": {"long": 9.993682, "lat": 53.551086},
}
