""""
Plotte Boxplot von Einstrahlungswerten eines Standorts
"""
import argparse

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.preprocessor import Preprocessor
from src.data.settings import locations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-x", "--location", help="Location", type=str, default="München"
    )
    args = parser.parse_args()

    location = args.location

    p = Preprocessor()
    df = p.load_locations()

    data = {}
    data_decomposed = {}
    for loc in locations:
        data[loc] = p.split_by_location(df, loc, "GlobInc")

    params = {
        "figure.figsize": (15, 10),
        "axes.labelsize": "16",
        "xtick.labelsize": "14",
        "ytick.labelsize": "14",
    }
    pylab.rcParams.update(params)

    # Boxplot normale Werte

    data[location]["Year"] = data[location].index.year
    data[location].index = pd.to_datetime(data[location].index)
    data[location]["Quarter"] = data[location].index.quarter
    data[location]["Quartal"] = "Q" + data[location]["Quarter"].astype(str)
    ax = sns.boxplot(
        x="Quartal",
        y="GlobInc",
        data=data[location].mask(data[location] == 0),
        palette="Set3",
        whis=1.5,
    )
    ax.set_ylabel("W/m²")
    ax.set_yticks([0, 200, 400, 600, 800, 1000, 1200, 1400])
    plt.show()
