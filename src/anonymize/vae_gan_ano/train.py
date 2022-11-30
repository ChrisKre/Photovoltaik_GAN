import argparse
import os
import sys

from src.anonymize.vae_gan_ano.model import VAEGan

root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(os.path.join(root))

from src.data.dataloader import DataLoaderEncSing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-x", "--location", help="Location", type=str, default="München"
    )
    parser.add_argument(
        "-xn", "--locationnorm", help="Location", type=str, default="Koethen"
    )
    parser.add_argument(
        "-s",
        "--season_decomp",
        help="Zerlegte Zeitreihendaten",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "-i",
        "--sample_interval",
        help="Klassifikation Evaluierung nach jedem interval",
        type=int,
        default=500,
    )
    parser.add_argument(
        "-e", "--epochs", help="Anzahl Trainingsdurchläufe", type=int, default=100000
    )
    args = parser.parse_args()

    x = args.location
    y = args.locationnorm
    sd = args.season_decomp

    epochs = args.epochs
    sample_interval = args.sample_interval
    dl = DataLoaderEncSing(x, season_decomp=sd)
    training_dataset, test_dataset = dl.get_train_test_data()

    dl = DataLoaderEncSing(y, season_decomp=sd)
    training_dataset_norm, _ = dl.get_train_test_data()
    timesteps, n_features = dl.get_dataset_shapes()
    test_dataset = test_dataset.batch(len(test_dataset))

    vaeGan = VAEGan(
        timesteps,
        n_features,
        3,
        location=x,
        season_decomp=sd,
        norm_location=y,
        disc_class=True,
    )
    vaeGan.train(
        training_dataset,
        norm_dataset=training_dataset_norm,
        epochs=epochs,
        sample_interval=sample_interval,
        test_dataset=test_dataset,
    )
