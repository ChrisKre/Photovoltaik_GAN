""""
Plotte Jahresverlauf von synthetischen Daten eines Standorts von VAE-GAN Modell
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

from src.data.dataloader import DataLoader
from src.data.settings import processed_dir
from src.utils import get_synth_dir, get_data_dir

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
        data_dir_fake = get_synth_dir("vae_gan", sd=False)
        data_dir_fake = os.path.join(data_dir_fake, location)
        fname_fake = os.path.join(data_dir_fake, os.listdir(data_dir_fake)[0])
    fake = dl.load_to_dataframe(fname_fake, season_decomp=False)

    data_dir_real = get_data_dir(processed_dir, scaled=scaled, sd=False)
    fname_real = os.path.join(data_dir_real, f"{location}.csv")
    real = dl.load_to_dataframe(fname_real, season_decomp=False, test_year=2016)

    index = pd.date_range(
        start=pd.Timestamp(year=2016, month=1, day=1, hour=0),
        end=pd.Timestamp(year=2016, month=12, day=31, hour=23),
        freq="1h",
    )
    df = pd.DataFrame(
        {"real": real.values.flatten(), "synth": fake.values.flatten()}, index=index
    )
    fig, ax = plt.subplots(nrows=5, figsize=(20, 15))
    ax[0].plot(df["real"], label="Real")
    ax[0].plot(df["synth"], label="Synth")
    ax[1].plot(df["real"].iloc[:500], label="Real")
    ax[1].plot(df["synth"].iloc[:500], label="Synth")
    ax[2].plot(df["real"].iloc[2500:3000], label="Real")
    ax[2].plot(df["synth"].iloc[2500:3000], label="Synth")
    ax[3].plot(df["real"].iloc[4500:5000], label="Real")
    ax[3].plot(df["synth"].iloc[4500:5000], label="Synth")
    ax[4].plot(df["real"].iloc[6500:7000], label="Real")
    ax[4].plot(df["synth"].iloc[6500:7000], label="Synth")
    plt.tight_layout()
    plt.legend()
    plt.show()
