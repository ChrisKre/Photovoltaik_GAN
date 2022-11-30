import argparse
import os.path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.settings import ano_dir, processed_dir, classification_labels
from src.classificator import Classificator

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
    locations = ["Madrid", "Le_Havre", "München"]

    fname_dir = os.path.join(
        ano_dir, "vae_gan", "normal", "disc_class", f"{location}_zu_{norm_location}"
    )
    fname_location = os.listdir(fname_dir)[0]
    fname_location = os.path.join(fname_dir, fname_location)

    df_train = pd.read_csv(fname_location, index_col=0)
    df_train.index = pd.to_datetime(df_train.index)
    df_train["target"] = classification_labels[norm_location]
    # df_train = df_location[df_location.index.year == 2016]

    fname_location = os.path.join(processed_dir, "scaled", f"{norm_location}.csv")
    df_norm_location = pd.read_csv(fname_location, index_col=0)
    df_norm_location.index = pd.to_datetime(df_norm_location.index)
    df_norm_location["target"] = classification_labels[norm_location]
    df_test = df_norm_location[df_norm_location.index.year == 2016]

    for location_next in locations:
        fname_location = os.path.join(processed_dir, "scaled", f"{location_next}.csv")
        df_loc_next = pd.read_csv(fname_location, index_col=0)
        df_loc_next.index = pd.to_datetime(df_loc_next.index)
        df_loc_next["target"] = classification_labels[location_next]
        # df_loc_next_train = df_loc_next[df_loc_next.index.year == 2016]
        df_loc_next_train = df_loc_next[df_loc_next.index.year == 2015]
        df_loc_next_test = df_loc_next[df_loc_next.index.year == 2016]

        df_train, df_test = pd.concat(
            [df_train, df_loc_next_train], ignore_index=True
        ), pd.concat([df_test, df_loc_next_test], ignore_index=True)

    train_target = tf.keras.utils.to_categorical(
        df_train["target"], num_classes=len(df_train["target"].unique())
    )
    test_target = tf.keras.utils.to_categorical(
        df_test["target"], num_classes=len(df_test["target"].unique())
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

    file_dir = os.path.join(
        "ano", "vae_gan", "train_synth_test_real", f"{location}_zu_{norm_location}"
    )
    cl = Classificator(
        input_dim=24, num_feat=1, num_classes=4, file_dir=file_dir, season_decomp=False
    )
    cl.train(training_dataset, epochs=100)
    cl.eval(test_dataset, ["Madrid", "Koethen", "München", "Le_Havre"])
