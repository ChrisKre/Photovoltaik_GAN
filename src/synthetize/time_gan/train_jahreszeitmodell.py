"""
Training eines TimeGAN-Modells für einen Standort und eine bestimmte Jahreszeit
Das Model wird gespeichert unter 'models/synthetize/time_gan/{normal bzw. season_decompose}/{season}/{location}'
"""

import argparse
import os
import sys
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(os.path.join(root))

from src.data.dataloader import DataLoaderEncSing
from src.synthetize.time_gan.model import TimeGAN
from src.utils import check_make_dir, get_model_dir
from src.synthetize.evaluate.utils import sample_evaluation

# Define names for model-parameters, which is needed for namedtuple and GAN
_model_parameters = [
    "batch_size",
    "lr",
    "beta1",
    "beta2",
    "layers_dim",
    "noise_dim",
    "n_cols",
]
_model_parameters_df = [128, 1e-4, (None, None), 128, 264, None, None, None, 1, None]

_train_parameters = ["cache_prefix", "label_dim", "epochs", "sample_interval", "labels"]

ModelParameters = namedtuple("ModelParameters", _model_parameters)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-x", "--location", help="Location", type=str, default="München"
    )
    parser.add_argument(
        "-s",
        "--season_decomp",
        help="Zerlegte Zeitreihendaten",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "-a",
        "--season",
        help="0: frühling, 1:sommer, 2:herbst, 3:winter",
        type=int,
        default=3,
    )
    args = parser.parse_args()

    # Set Parameter
    scaled = True
    max_train_steps = 100000
    train_steps = 10000
    x = args.location
    season_decomp = args.season_decomp
    season = args.season

    # Load trainingdata
    dl = DataLoaderEncSing(x, season_decomp=season_decomp, season=season)
    train_dataset, test_dataset = dl.get_train_test_data()
    timesteps, n_features = dl.get_dataset_shapes()

    # Set GAN trainingparameter
    gan_args = ModelParameters(
        batch_size=128,
        lr=5e-4,
        noise_dim=32,
        layers_dim=128,
        beta1=None,
        beta2=None,
        n_cols=None,
    )

    # Define Save dir for model
    save_dir = get_model_dir("time_gan", season_decomp)
    if season == 0:
        save_dir = os.path.join(save_dir, "frühling")
    elif season == 1:
        save_dir = os.path.join(save_dir, "sommer")
    elif season == 2:
        save_dir = os.path.join(save_dir, "herbst")
    else:
        save_dir = os.path.join(save_dir, "winter")
    save_dir = os.path.join(save_dir, x)

    weights_dir = os.path.join(save_dir, "weights")
    check_make_dir(weights_dir)
    model_fname = os.path.join(save_dir, "model.pkl")
    result_fname = os.path.join(save_dir, "results.csv")

    # Instanziate Model
    synth = TimeGAN(
        model_parameters=gan_args,
        hidden_dim=12,
        seq_len=timesteps,
        n_seq=n_features,
        gamma=1,
    )

    # Define Metricsname for evaluation during training
    metrics_names = [
        "cl_loss_loc",
        "cl_accuracy_loc",
        "cl_precision_loc",
        "cl_recall_loc",
    ]
    df_result = pd.DataFrame()

    # Define test_dataset for evaluation during training
    sample_size = len(test_dataset)
    test_dataset = test_dataset.batch(len(test_dataset))

    # Load classficationmodel for evaluation during training
    eval_classificator_dir = get_model_dir(sd=season_decomp)
    eval_classificator_model = tf.keras.models.load_model(
        os.path.join(eval_classificator_dir, "model")
    )

    # Reshape trainingdataset accoridng to season_decomp
    if season_decomp:
        train_dataset = np.array([i.numpy() for i in train_dataset])
    else:
        train_dataset = np.array([i[0] for i in train_dataset])

    # Start training, each {train_steps} make classification for location
    for i in range(0, max_train_steps, train_steps):
        # Run training
        synth.train(train_dataset, train_steps=train_steps, save_dir=save_dir)

        # Create evaluation dataset
        synth_data = synth.sample(sample_size)

        # Evaluate with pretrained classificator
        eval_result = sample_evaluation(eval_classificator_model, synth_data, x)

        # Log evaluation results
        df_result = pd.concat(
            [df_result, pd.DataFrame([eval_result], index=[i], columns=metrics_names)]
        )
        rstring = f"Epoche {i}\t"
        for name, value in zip(metrics_names, eval_result):
            rstring += f"{name}: {value} \t"
        print(rstring)
        weights_fname = os.path.join(weights_dir, "weights_{}.h5".format(i))

        # Save generator weights
        synth.generator.save_weights(weights_fname)

    # Plot and save classification evaluation results
    check_make_dir(model_fname, file=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    for col, ax in zip(df_result.columns, axes.flatten()):
        df_result.reset_index().plot(ax=ax, x="index", y=col)

    df_result.to_csv(result_fname)
    fig_fname = os.path.join(save_dir, "results.png")
    plt.savefig(fig_fname)
