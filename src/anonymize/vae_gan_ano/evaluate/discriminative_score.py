"""
Berechne discriminativen score zwischen anonymisierten Daten und Daten des Normstandorts
"""
import argparse
import os.path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.classificator.classificator import Classificator
from src.data.settings import ano_dir, processed_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--location", help="Location", type=str, default="Madrid")
    parser.add_argument(
        "-y", "--norm_location", help="Location", type=str, default="Koethen"
    )
    args = parser.parse_args()

    # Set Paramter
    location = args.location
    norm_location = args.norm_location

    # Load data
    fdir_ano = os.path.join(
        ano_dir, "vae_gan", "normal", f"{location}_zu_{norm_location}"
    )
    fname_ano = os.path.join(fdir_ano, os.listdir(fdir_ano)[0])
    df_location = pd.read_csv(fname_ano, index_col=0)
    df_location.index = pd.to_datetime(df_location.index)
    df_location["target"] = 0
    df_location_train = df_location.sample(frac=0.9)
    df_location_test = df_location.drop(df_location_train.index)

    fname_location = os.path.join(processed_dir, "scaled", f"{location}.csv")
    df_norm_location = pd.read_csv(fname_location, index_col=0)
    df_norm_location.index = pd.to_datetime(df_norm_location.index)
    df_norm_location["target"] = 1
    df_norm_location = df_norm_location[df_norm_location.index.year == 2016]
    df_norm_location_train = df_norm_location.sample(frac=0.9)
    df_norm_location_test = df_norm_location.drop(df_norm_location_train.index)

    # Create Datasets
    df_train, df_test = pd.concat(
        [df_location_train, df_norm_location_train], ignore_index=True
    ), pd.concat([df_location_test, df_norm_location_test], ignore_index=True)

    train_target = tf.cast(
        np.asarray(df_train["target"].values).reshape((-1, 1)), tf.int16
    )
    test_target = tf.cast(
        np.asarray(df_test["target"].values).reshape((-1, 1)), tf.int16
    )

    training_dataset = tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(
                np.expand_dims(
                    df_train[df_train.columns.difference(["target"])].values, axis=2
                ),
                tf.float32,
            ),
            train_target,
        )
    )

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(
                np.expand_dims(
                    df_test[df_test.columns.difference(["target"])].values, axis=2
                ),
                tf.float32,
            ),
            test_target,
        )
    )

    # Define save_dir
    file_dir = os.path.join(
        "ano", "vae_gan", "disc_score", f"{location}_zu_{norm_location}"
    )
    cl = Classificator(
        input_dim=24,
        num_feat=1,
        num_classes=2,
        file_dir=file_dir,
        season_decomp=False,
    )

    # Train and evaluate
    cl.train(training_dataset, epochs=100)
    cl.eval(test_dataset, [location, norm_location])
