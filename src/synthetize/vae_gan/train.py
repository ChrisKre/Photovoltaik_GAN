"""
Training des VAE-GANs zur Synthetisierung eines Standorts.
Das Model wird gespeichert unter 'models/synthetize/vae_gan/{normal bzw. season_decompose}/{location}'
Setze Parameter
 location: Bestimme den Standort der Trainingsdaten
 season_decomp: Ob das trainierte Model mit zerlegten oder normalen Daten trainiert werden soll
Während dem Training wird nach jedem {sample_interval} eine Evaluierung mit einem vortrainierten Klassifikator unternommen.
Dieser muss bereit liegen. Daher: src/classificator/train_classificator.py zunächst ausführen
"""

import argparse
import os
import sys

root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(os.path.join(root))

from src.data.dataloader import DataLoaderEncSing
from src.synthetize.vae_gan.model import VAEGan

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

    # Set Parameter
    x = args.location
    sd = args.season_decomp
    epochs = args.epochs
    sample_interval = args.sample_interval

    # Load training data
    dl = DataLoaderEncSing(x, season_decomp=sd)
    training_dataset, test_dataset = dl.get_train_test_data()
    timesteps, n_features = dl.get_dataset_shapes()
    test_dataset = test_dataset.batch(len(test_dataset))

    # Run Training
    vaeGan = VAEGan(timesteps, n_features, latent_depth=3, location=x, season_decomp=sd)
    vaeGan.train(
        training_dataset,
        epochs=epochs,
        sample_interval=sample_interval,
        test_dataset=test_dataset,
    )
