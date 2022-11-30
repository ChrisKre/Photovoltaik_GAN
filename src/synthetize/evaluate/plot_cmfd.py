"""
Erstelle plots von Wahrscheinlichkeitsverteilungen.
Diese werden defalutmäßig im selben Ordner gespeichert
"""

import argparse
import os.path

from src.anonymize.vae_gan_ano.evaluate.pdf_cdf import plot_pdf_cdf, plot_pds
from src.data.dataloader import DataLoader
from src.data.settings import processed_dir
from src.utils import check_make_dir, get_synth_dir, get_data_dir

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
        "-sf", "--save_file_dir", help="Save Filedir", type=str, default="pdf_cdf"
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
    gan_type = args.gan
    test_year = 2016

    # Load fake data, if no file_name given take file from model folder
    dl = DataLoader()
    if fname_fake is None:
        data_dir_fake = get_synth_dir(gan_type, season_decomp)
        data_dir_fake = os.path.join(data_dir_fake, location)
        fname_fake = os.path.join(data_dir_fake, os.listdir(data_dir_fake)[0])
    fake = dl.load_to_dataframe(fname_fake, season_decomp=season_decomp)

    # Load real data
    data_dir_real = get_data_dir(processed_dir, scaled=scaled, sd=season_decomp)
    fname_real = os.path.join(data_dir_real, f"{location}.csv")
    real = dl.load_to_dataframe(
        fname_real, season_decomp=season_decomp, test_year=test_year
    )
    fake = fake.iloc[-len(real) :]

    # Set save dir
    f_dir = os.path.join(save_file_dir, f"{location}")
    check_make_dir(f_dir)

    # Plot pdf and cdf
    fname_pdf = os.path.join(f_dir, "pdf_cdf.png")
    plot_pdf_cdf(real.values.flatten(), fake.values.flatten(), fname_pdf, synth=False)

    # Plot pds
    fname_pds = os.path.join(f_dir, "pds.png")
    plot_pds(real.values.flatten(), fake.values.flatten(), fname_pds)
