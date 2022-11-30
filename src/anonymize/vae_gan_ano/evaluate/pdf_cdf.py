import argparse
import os.path

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

from src.data.dataloader import DataLoader
from src.data.settings import ano_dir, processed_dir
from src.utils import check_make_dir


def plot_pdf_cdf(
    location: np.ndarray, norm_location: np.ndarray, fname: str, synth: True
):
    """
    Plot the PDF and CDF Probability distribution for given data
    :param df_location:
    :param df_norm_location:
    :param fname:
    :return:
    """
    # getting data of the histogram
    count_norm, bins_count_norm = np.histogram(norm_location, bins=24)
    # finding the PDF of the histogram using count values
    pdf_norm = count_norm / sum(count_norm)
    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf_norm = np.cumsum(pdf_norm)
    # getting data of the histogram
    count_ano, bins_count_ano = np.histogram(location, bins=24)
    # finding the PDF of the histogram using count values
    pdf_ano = count_ano / sum(count_ano)
    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf_ano = np.cumsum(pdf_ano)
    # plotting PDF and CDF
    if synth:
        label_1 = "Real"
        label_2 = "Synth"
    else:
        label_1 = "Norm"
        label_2 = "Ano"

    plt.plot(bins_count_norm[1:], pdf_norm, color="blue", label=f"PDF {label_1}")
    plt.plot(bins_count_norm[1:], cdf_norm, color="green", label=f"CDF {label_1}")
    plt.plot(bins_count_ano[1:], pdf_ano, color="yellow", label=f"PDF {label_2}")
    plt.plot(bins_count_ano[1:], cdf_ano, color="orange", label=f"CDF {label_2}")
    plt.legend()
    plt.savefig(fname)


def plot_pds(location, norm_location, fname):
    """
    Plot the PDS Probability distribution for given data
    :param location:
    :param norm_location:
    :param fname:
    :return:
    """
    # set random state for reproducibility
    freqs_t, psd_t = signal.welch(norm_location)
    freqs_v, psd_v = signal.welch(location)
    plt.figure(figsize=(5, 4))
    plt.semilogx(freqs_t, psd_t)
    plt.semilogx(freqs_v, psd_v)
    plt.title("PSD: power spectral density")
    plt.xlabel("Frequency")
    plt.ylabel("Power")
    plt.tight_layout()
    plt.savefig(fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-y", "--norm_location", help="Location", type=str, default="Koethen"
    )

    args = parser.parse_args()
    locations = ["Le_Havre", "Madrid", "München"]
    norm_location = args.norm_location
    dl = DataLoader()
    test_year = 2016
    for location in locations:
        fdir_ano = os.path.join(ano_dir, "vae_gan", f"{location}_zu_{norm_location}")
        fname_ano = os.path.join(fdir_ano, os.listdir(fdir_ano)[0])

        df_location = dl.load_to_dataframe(fname_ano, season_decomp=False)
        df_location = df_location.values.flatten()

        fname_location = os.path.join(processed_dir, "scaled", f"{norm_location}.csv")
        df_norm_location = dl.load_to_dataframe(
            fname_location, season_decomp=False, test_year=test_year
        )
        df_norm_location = df_norm_location.values.flatten()

        fdir = os.path.join('pdf_cdf', f"{location}_zu_{norm_location}")
        check_make_dir(fdir)

        fname_pdf = os.path.join(fdir, "pdf_cdf.png")
        plot_pdf_cdf(df_location, df_norm_location, fname_pdf, synth=False)

        fname_pds = os.path.join(fdir, "pds.png")
        plot_pds(df_location, df_norm_location, fname_pds)
