"""
Erstelle synthetische Datensätze für das VAE-GAN.
Setze Parameter
 location: Das trainierte Model für den angegeben Standort
 weights: Für jedes angegebene Gewicht wird ein Datensatz erstellt und 'data/synth/vaegan/normal bzw season_decomposed/location/weight' gespeichert
 season_decomp: Ob das trainierte Model mit zerlegten oder normalen Daten trainiert wurde
"""

import argparse
import os
import sys

root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(os.path.join(root))
from src.data.dataloader import DataLoader
from src.synthetize.evaluate.utils import synth_to_df
from src.synthetize.vae_gan.model import VAEGan
from src.utils import check_make_dir, get_model_dir, get_synth_dir, get_data_dir
from src.data.settings import processed_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-x", "--location", help="Location", type=str, default="Koethen"
    )
    parser.add_argument(
        "-w",
        "--weights",
        help="Which weights of the Model",
        type=list,
        default=[500],
    )
    parser.add_argument(
        "-sd",
        "--season_decomp",
        help="Make seasonal decomposition",
        type=bool,
        default=False,
    )
    args = parser.parse_args()

    # Set Parameter
    x = args.location
    weights = args.weights
    season_decomp = args.season_decomp

    # Get file_dir of Model
    save_dir = get_model_dir("vae_gan", season_decomp, ano=False)
    save_dir = os.path.join(save_dir, f"{x}")

    # Load real data to reconstruct
    data_dir_real = get_data_dir(processed_dir, scaled=True, sd=season_decomp)
    fname_real = os.path.join(data_dir_real, f"{x}.csv")
    dl = DataLoader()

    # Extract only test year 2016
    real = dl.load_to_numpy(fname_real, season_decomp=False)
    real = real[-366:]

    timesteps, n_features = real.shape[1], real.shape[2]
    # Iterate over weights and create synth dataset for each weight
    for weight in weights:
        # Load model with corresponding weight
        weights_dir = os.path.join(save_dir, "weights")
        vaeGan = VAEGan(
            timesteps,
            n_features,
            3,
            location=x,
            season_decomp=season_decomp,
            weights=weight,
            weights_dir=weights_dir,
        )

        # Create dataset
        test_batch_fake, test_dataset_batch_random = vaeGan.sample_x(real)
        df = synth_to_df(season_decomp, test_batch_fake.numpy())

        # Define Save dir for dataset
        data_dir = get_synth_dir("vae_gan", season_decomp)
        fname = os.path.join(data_dir, x, f"{weight}.csv")
        check_make_dir(fname, True)
        # Save dataset
        df.to_csv(fname)
