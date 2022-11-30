"""
Erstelle synthetische Datensätze für das TimeGAN.
Setze Parameter
 location: Das trainierte Model für den angegeben Standort
 weights: Für jedes angegebene Gewicht wird ein Datensatz erstellt und 'data/synth/time_gan/{normal bzw season_decomposed}/{location}/{weight}' gespeichert
 season_decomp: Ob das trainierte Model mit zerlegten oder normalen Daten trainiert wurde
"""

import argparse
import os
import sys

root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(os.path.join(root))

from src.synthetize.time_gan.train import ModelParameters
from src.utils import check_make_dir, get_model_dir, get_synth_dir
from src.synthetize.evaluate.utils import synth_to_df
from src.synthetize.time_gan.model import TimeGAN

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-x", "--location", help="Location", type=str, default="München"
    )
    parser.add_argument(
        "-sd",
        "--season_decomp",
        help="Make seasonal decomposition",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "-w",
        "--weights",
        help="Which weights of the Model",
        type=list,
        default=[50],
    )
    args = parser.parse_args()

    # Set Parameter
    x = args.location
    season_decomp = args.season_decomp
    weights = args.weights

    # Set Number of Features
    n_seq = 1 if not season_decomp == True else 4

    # Get file_dir of Model
    save_dir = get_model_dir("time_gan", season_decomp)
    save_dir = os.path.join(save_dir, x)

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
        synth_data = synth.sample(365)
        df = synth_to_df(season_decomp, synth_data)

        # Define Save dir for dataset
        data_dir = get_synth_dir("time_gan", season_decomp)
        fname = os.path.join(data_dir, x, f"{weight}.csv")

        # Save dataset
        check_make_dir(fname, True)
        df.to_csv(fname)
