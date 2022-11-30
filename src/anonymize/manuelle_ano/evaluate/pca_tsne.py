import argparse
import os.path

import numpy as np
import pandas as pd

from src.data.settings import ano_dir, processed_dir
from src.utils import check_make_dir
from src.vis.visual_helper import plot_pca, plot_tsne

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-x", "--location", help="Location", type=str, default="München"
    )
    parser.add_argument(
        "-y", "--norm_location", help="Location", type=str, default="Koethen"
    )
    args = parser.parse_args()

    location = args.location
    norm_location = args.norm_location

    fname_location = os.path.join(ano_dir, "manuelle", f"{location}.csv")
    df_location = pd.read_csv(fname_location, index_col=0)
    df_location.index = pd.to_datetime(df_location.index)
    df_location_test = df_location[df_location.index.year == 2016]
    df_location_test["target"] = "ano"

    fname_location = os.path.join(processed_dir, f"{norm_location}.csv")
    df_norm_location = pd.read_csv(fname_location, index_col=0)
    df_norm_location.index = pd.to_datetime(df_norm_location.index)
    df_norm_location_test = df_norm_location[df_norm_location.index.year == 2016]
    df_norm_location_test["target"] = "norm"

    f_dir = os.path.join("pca_tsne", f"{location}_zu_{norm_location}")

    target = pd.concat([df_norm_location_test["target"], df_location_test["target"]])
    data_reduced = np.concatenate(
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
    plot_pca(data_reduced, target.values, fname_pca)

    fname_tsne = os.path.join(f_dir, "tsne.png")
    plot_tsne(data_reduced, target.values, fname_tsne)
