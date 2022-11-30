"""
Erstelle Plot für Korrelationsmatrix für echte und synthetische Daten
"""
import argparse
import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, pylab

from src.data.dataloader import DataLoader
from src.data.settings import processed_dir, synth_dir


def default_fname(location):
    """
    Return default-name for synth data
    :param location:
    :return:
    """
    dir = os.path.join(synth_dir, "vae_gan", "normal", location)
    fname = os.path.join(dir, os.listdir(dir)[0])
    return fname


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k", "--fname_koethen", type=str, default=default_fname("Koethen")
    )
    parser.add_argument(
        "-l", "--fname_le_havre", type=str, default=default_fname("Le_Havre")
    )
    parser.add_argument(
        "-mu", "--fname_muenchen", type=str, default=default_fname("München")
    )
    parser.add_argument(
        "-ma", "--fname_madrid", type=str, default=default_fname("Madrid")
    )
    args = parser.parse_args()

    # Set Parameter
    test_year = 2016
    dl = DataLoader()
    fname_k = os.path.join(processed_dir, "scaled", "Koethen.csv")
    fname_l = os.path.join(processed_dir, "scaled", "Le_Havre.csv")
    fname_ma = os.path.join(processed_dir, "scaled", "Madrid.csv")
    fname_mu = os.path.join(processed_dir, "scaled", "München.csv")

    # Load real_data
    df_k = dl.load_to_dataframe(fname_k, season_decomp=False, test_year=test_year)
    df_l = dl.load_to_dataframe(fname_l, season_decomp=False, test_year=test_year)
    df_ma = dl.load_to_dataframe(fname_ma, season_decomp=False, test_year=test_year)
    df_mu = dl.load_to_dataframe(fname_mu, season_decomp=False, test_year=test_year)

    # Load synth data vae_gan
    df__vaegan_k = pd.read_csv(args.fname_koethen, index_col=0)
    df__vaegan_l = pd.read_csv(args.fname_le_havre, index_col=0)
    df__vaegan_mu = pd.read_csv(args.fname_muenchen, index_col=0)
    df__vaegan_ma = pd.read_csv(args.fname_madrid, index_col=0)

    # Concat to one time series
    df_k = df_k.values.flatten()
    df_l = df_l.values.flatten()
    df_mu = df_mu.values.flatten()
    df_ma = df_ma.values.flatten()
    len = len(df_l)

    df__vaegan_k = df__vaegan_k.values.flatten()[:len]
    df__vaegan_l = df__vaegan_l.values.flatten()[:len]
    df__vaegan_ma = df__vaegan_ma.values.flatten()[:len]
    df__vaegan_mu = df__vaegan_mu.values.flatten()[:len]

    # Create dataframe for easy correlation calculate accross columns
    df_vaegan = pd.DataFrame(
        {
            "Koethen": df_k,
            "Le Havre": df_l,
            "Madrid": df_ma,
            "München": df_mu,
            "Koethen VAE-GAN": df__vaegan_k,
            "Le Havre VAE-GAN": df__vaegan_l,
            "Madrid VAE-GAN": df__vaegan_ma,
            "München VAE-GAN": df__vaegan_mu,
        }
    )
    params = {
        "figure.figsize": (18, 18),
        "axes.labelsize": "16",
        "xtick.labelsize": "14",
        "ytick.labelsize": "14",
    }
    pylab.rcParams.update(params)

    # Calculate correlation for all columns
    vaegancorr = df_vaegan.corr()
    print(vaegancorr)
    # Create Plot
    sns.heatmap(
        vaegancorr,
        xticklabels=vaegancorr.columns.values,
        yticklabels=vaegancorr.columns.values,
    )
    plt.show()
