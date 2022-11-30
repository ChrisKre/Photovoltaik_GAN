"""
Erstelle plots von komprimierten Darstellungen.
Diese werden defalutmäßig im selben Ordner gespeichert
"""

import argparse
import os.path

import numpy as np
import pandas as pd

from src.data.dataloader import DataLoader
from src.data.settings import processed_dir
from src.utils import check_make_dir, get_synth_dir, get_data_dir
from src.vis.visual_helper import plot_pca, plot_tsne

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-x", "--location", help="Location", type=str, default="München"
    )
    parser.add_argument("-s", "--scaled", help="Scaled", type=bool, default=True)
    parser.add_argument(
        "-sd",
        "--season_decomp",
        help="Make seasonal decomposition",
        type=bool,
        default=False,
    )
    parser.add_argument("-g", "--gan", help="Gantype", type=str, default="vae_gan")
    parser.add_argument(
        "-sf", "--save_file_dir", help="Save Filedir", type=str, default="pca_tsne"
    )
    parser.add_argument(
        "-f", "--file_name", help="Filename of fake data", type=str, default=None
    )
    args = parser.parse_args()

    # Set Parameter
    location = args.location
    season_decomp = args.season_decomp
    scaled = args.scaled
    save_file_dir = args.save_file_dir
    fname_fake = args.file_name
    gan_type = args.gan
    test_year = 2016

    # Load fake data, if no file_name given take file from model folder
    dl = DataLoader()
    if fname_fake is None:
        data_dir_fake = get_synth_dir(gan_type, season_decomp)
        data_dir_fake = os.path.join(data_dir_fake, location)
        fname_fake = os.path.join(data_dir_fake, os.listdir(data_dir_fake)[0])
    fake = dl.load_to_dataframe(fname_fake, season_decomp=season_decomp)
    # Set Label for plot
    fake["target"] = "Synthetisch"

    # Load real data
    data_dir_real = get_data_dir(processed_dir, scaled=scaled, sd=season_decomp)
    fname_real = os.path.join(data_dir_real, f"{location}.csv")
    real = dl.load_to_dataframe(
        fname_real, season_decomp=season_decomp, test_year=test_year
    )
    # Set Label for plot
    real["target"] = "Original"
    fake = fake.iloc[-len(real) :]

    # Set save dir
    f_dir = os.path.join(save_file_dir, f"{location}")
    check_make_dir(f_dir)

    # Concat for plot
    target = pd.concat([real["target"], fake["target"]])

    data_reduced = np.concatenate(
        (
            real[real.columns.difference(["target"])].values,
            fake[fake.columns.difference(["target"])].values,
        ),
        axis=0,
    )

    # Plot pca
    fname_pca = os.path.join(f_dir, "pca.png")
    plot_pca(data_reduced, target.values, fname_pca)

    # Plot t-sne
    fname_tsne = os.path.join(f_dir, "tsne.png")
    plot_tsne(data_reduced, target.values, fname_tsne)
