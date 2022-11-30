"""
Durchführung einer Klassifikation von echten Daten und Training eines Klassifikators mit synthetischen Daten für alle Standorte.
bereitsteht. Ergebnisse werden defaultmäßig im selben ordner unter 'tstr' gespeichert
"""

import argparse

from src.synthetize.evaluate.classifier_evaluation import ClassifierEvaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
        "-sf", "--save_file_dir", help="Save Filedir", type=str, default="tstr"
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
    gan = args.gan

    if gan == "vae_gan":
        model = False
    else:
        model = True

    # Apply TSTR
    eval = ClassifierEvaluation(save_file_dir, scaled, season_decomp)
    eval.train_synth_test_real(time_gan=model)
