"""
Berechne discriminativen score
 -> Klassifikation zwischen echten (0) und synthetischen Daten (1) eines Standorts
"""
import argparse
import os

from src.data.dataloader import DataLoader
from src.data.settings import processed_dir
from src.utils import get_synth_dir, get_data_dir
from src.synthetize.evaluate.classifier_evaluation import ClassifierEvaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-x", "--location", help="Location", type=str, default="München"
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
        "-sf", "--save_file_dir", help="Save Filedir", type=str, default="disc_score"
    )
    parser.add_argument(
        "-f", "--file_name", help="Filename of fake data", type=str, default=None
    )
    args = parser.parse_args()

    location = args.location
    season_decomp = args.season_decomp
    scaled = args.scaled
    save_file_dir = args.save_file_dir
    fname_fake = args.file_name

    # Load synth data
    dl = DataLoader()
    if fname_fake is None:
        data_dir_fake = get_synth_dir("time_gan", season_decomp)
        data_dir_fake = os.path.join(data_dir_fake, location)
        fname_fake = os.path.join(data_dir_fake, os.listdir(data_dir_fake)[0])
    fake = dl.load_to_numpy(fname_fake, season_decomp=season_decomp)

    # Load real data
    data_dir_real = get_data_dir(processed_dir, scaled=scaled, sd=season_decomp)
    fname_real = os.path.join(data_dir_real, f"{location}.csv")
    real = dl.load_to_numpy(fname_real, season_decomp=season_decomp)

    # Train classifier for evaluation
    eval = ClassifierEvaluation(save_file_dir)
    eval.discriminative_score(real=real, fake=fake)
