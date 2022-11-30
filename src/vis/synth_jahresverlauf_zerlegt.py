"""
Plotted einen synthetischen zerlegten Jahresverlauf
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

from src.data.dataloader import DataLoader
from src.utils import get_synth_dir, seasonal_decomposed_cols

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-x", "--location", help="Location", type=str, default="Koethen"
    )
    parser.add_argument("-s", "--scaled", help="Scaled", type=bool, default=True)
    parser.add_argument(
        "-f", "--file_name", help="Filename of fake data", type=str, default=None
    )
    args = parser.parse_args()

    location = args.location
    scaled = args.scaled
    fname_fake = args.file_name

    dl = DataLoader()

    x = args.location

    if fname_fake is None:
        data_dir_fake = get_synth_dir("vae_gan", sd=True)
        data_dir_fake = os.path.join(data_dir_fake, location)
        fname_fake = os.path.join(data_dir_fake, os.listdir(data_dir_fake)[0])
    fake = dl.load_to_numpy(fname_fake, season_decomp=True)
    fake = fake.reshape((fake.shape[0] * fake.shape[1], 4))
    df_fake = pd.DataFrame(
        {
            seasonal_decomposed_cols[0]: fake[:, 0],
            seasonal_decomposed_cols[1]: fake[:, 1],
            seasonal_decomposed_cols[2]: fake[:, 2],
            seasonal_decomposed_cols[3]: fake[:, 3],
        }
    )
    df_fake["observed"] = df_fake.sum(1)

    f, axes = plt.subplots(5, 1, figsize=(18, 24), dpi=200)

    axes[0].plot(df_fake["observed"], color="green", label="Beobachtete Daten")
    axes[0].legend(loc="upper right", prop={"size": 16})
    # plotting daily seasonal component
    axes[1].plot(
        df_fake["seasonal_24"].iloc[:1000],
        linewidth=2,
        label="Tägliche saisonale Komponente",
    )
    # plotting weekly seasonal component
    axes[1].legend(loc="upper right", prop={"size": 16})
    axes[2].plot(
        df_fake["seasonal_8760"].iloc[:8760],
        color="orange",
        label="Jährliche saisonale Komponente",
    )
    axes[2].legend(loc="upper right", prop={"size": 16})
    # plotting trend component
    axes[3].plot(df_fake["trend"], color="r", label="Trend Komponente")
    axes[3].legend(loc="upper right", prop={"size": 16})
    # plotting yearly seasonality
    axes[4].plot(df_fake["resid"], color="green", label="Rauschkomponente")
    axes[4].legend(loc="upper right", prop={"size": 16})
    # plotting residual of decomposition
    for a in axes:
        a.set_ylabel("W/m²", fontsize=14)
        a.tick_params(axis="y", labelsize=12)
        a.tick_params(axis="x", labelsize=12)
    plt.tight_layout()
    df_fake.plot(subplots=True)
    plt.show()

    index = pd.date_range(
        start=pd.Timestamp(year=2016, month=1, day=1, hour=0),
        end=pd.Timestamp(year=2016, month=12, day=31, hour=23),
        freq="1h",
    )
