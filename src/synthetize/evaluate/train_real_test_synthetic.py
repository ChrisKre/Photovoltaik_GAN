"""
Durchführung einer Klassifikation von synthetischen Daten mit bereits trainiertem Klassifikator für alle Standorte.
Dazu muss zunächst 'src/classificator/train_classificator.py' ausgeführt werden, so dass der trainierte Klassifikatoren
bereitsteht. Ergebnisse werden defaultmäßig im selben ordner unter 'trts/{location}' gespeichert
"""

import argparse
import os

from src.data.dataloader import DataLoader
from src.data.settings import classification_labels
from src.synthetize.evaluate.classifier_evaluation import ClassifierEvaluation
from src.utils import get_synth_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-x", "--location", help="Location", type=str, default="Le_Havre"
    )
    parser.add_argument("-s", "--scaled", help="Scaled", type=bool, default=True)
    parser.add_argument(
        "-sd",
        "--season_decomp",
        help="Make seasonal decomposition",
        type=bool,
        default=False,
    )
    parser.add_argument("-g", "--gan", help="Gantype", type=str, default="vae_gan")
    parser.add_argument(
        "-sf", "--save_file_dir", help="Save Filedir", type=str, default="trts"
    )
    parser.add_argument(
        "-f", "--file_name", help="Filename of fake data", type=str, default=None
    )
    args = parser.parse_args()

    # Set Parameter
    location = args.location
    season_decomp = args.season_decomp
    scaled = args.scaled
    save_file_dir = args.save_file_dir
    fname_fake = args.file_name

    # Load fake data, if no file_name given take file from model folder
    dl = DataLoader()
    if fname_fake is None:
        data_dir_fake = get_synth_dir("time_gan", season_decomp)
        data_dir_fake = os.path.join(data_dir_fake, location)
        fname_fake = os.path.join(data_dir_fake, os.listdir(data_dir_fake)[0])
    fake = dl.load_to_numpy(fname_fake, season_decomp=season_decomp)

    # Make Evaluation
    eval = ClassifierEvaluation(save_file_dir)
    # Set label according to settings
    y_true = classification_labels[location]
    eval.train_real_test_synth(fake, location)
