"""
Erstelle synthetische Datensätze für das VAE-GAN-Jahreszeitmodell.
Setze Parameter
 location: Das trainierte Model für den angegeben Standort
 locationnorm: Den Normstandort, üblicherweise Koethen
 weights: Für jedes angegebene Gewicht wird ein Datensatz erstellt und 'data/synth/vaegan/{normal bzw season_decomposed}/{season}/{location}/{weight}' gespeichert
 season_decomp: Ob das trainierte Model mit zerlegten oder normalen Daten trainiert wurde- Default ist False
"""

import argparse
import os
import sys

from src.synthetize.evaluate.utils import synth_to_df
from src.synthetize.time_gan.model import TimeGAN

root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(os.path.join(root))

from src.synthetize.time_gan.train import ModelParameters
from src.utils import check_make_dir, get_model_dir, get_synth_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-x", "--location", help="Location", type=str, default="Koethen"
    )
    parser.add_argument(
        "-w", "--weights", help="Which weights of the Model", type=list, default=[40000]
    )
    parser.add_argument("-s", "--season", help="Which Season", type=int, default=0)
    args = parser.parse_args()

    # Set Parameter
    x = args.location
    season = args.season
    weights = args.weights

    # Set Number of Features

    n_seq = 1
    # Get file_dir of Model
    save_dir = get_model_dir("time_gan", False)

    # Determine season
    if season == 0:
        s_string = "frühling"
    elif season == 1:
        s_string = "sommer"
    elif season == 2:
        s_string = "herbst"
    else:
        s_string = "winter"
    save_dir = os.path.join(save_dir, s_string, x)

    # Iterate over weights and create synth dataset for each weight
    for weight in weights:
        # Load model with corresponding weight
        weights_dir = os.path.join(save_dir, "weights")
        weights_fname = os.path.join(weights_dir, "weights_{}.h5".format(str(weight)))
        gan_args = ModelParameters(
            batch_size=1,
            lr=5e-4,
            noise_dim=32,
            layers_dim=128,
            beta1=None,
            beta2=None,
            n_cols=None,
        )

        synth = TimeGAN(
            model_parameters=gan_args, hidden_dim=12, seq_len=24, n_seq=n_seq, gamma=1
        )
        synth.generator.load_weights(weights_fname)

        # Create dataset
        synth_data = synth.sample(4383)
        df = synth_to_df(False, synth_data)

        # Define Save dir for dataset
        data_dir = get_synth_dir("time_gan", False)
        fname = os.path.join(data_dir, s_string, x, f"{weight}.csv")

        # Save dataset
        check_make_dir(fname, True)
        df.to_csv(fname)
