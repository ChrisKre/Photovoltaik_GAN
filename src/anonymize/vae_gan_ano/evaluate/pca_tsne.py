import argparse
import os.path

import numpy as np
import pandas as pd

from src.data.dataloader import DataLoader
from src.data.settings import ano_dir, processed_dir
from src.utils import check_make_dir
from src.vis.visual_helper import plot_pca, plot_tsne

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-x", "--location", help="Location", type=str, default="Le_Havre"
    )
    parser.add_argument(
        "-y", "--norm_location", help="Location", type=str, default="Koethen"
    )

    args = parser.parse_args()

    location = args.location
    norm_location = args.norm_location
    test_year = 2016

    dl = DataLoader()

    fdir_location = os.path.join(ano_dir, "vae_gan", f"{location}_zu_{norm_location}")
    fname_location = os.path.join(fdir_location, os.listdir(fdir_location)[0])
    df_location_test = dl.load_to_dataframe(fname_location, season_decomp=False)
    df_location_test["target"] = "ano"

    fname_norm_location = os.path.join(processed_dir, "scaled", f"{norm_location}.csv")
    df_norm_location_test = dl.load_to_dataframe(
        fname_norm_location, season_decomp=False, test_year=test_year
    )
    df_norm_location_test["target"] = "norm"

    f_dir = os.path.join("pca_tsne", f"{location}_zu_{norm_location}")
    check_make_dir(f_dir)

    target = pd.concat([df_norm_location_test["target"], df_location_test["target"]])

    data = np.concatenate(
        (
            df_norm_location_test[
                df_norm_location_test.columns.difference(["target"])
            ].values,
            df_location_test[df_location_test.columns.difference(["target"])].values,
        ),
        axis=0,
    )

    fname_pca = os.path.join(f_dir, "pca.png")
    check_make_dir(fname_pca, True)
    plot_pca(data, target.values, fname_pca)

    fname_tsne = os.path.join(f_dir, "tsne.png")
    plot_tsne(data, target.values, fname_tsne)
