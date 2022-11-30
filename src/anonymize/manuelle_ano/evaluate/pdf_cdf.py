import argparse
import os.path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal

from src.data.settings import ano_dir, processed_dir
from src.utils import check_make_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-y", "--norm_location", help="Location", type=str, default="Koethen"
    )

    args = parser.parse_args()
    locations = ["Le_Havre", "Madrid", "München"]
    norm_location = args.norm_location
    for location in locations:
        fname_location = os.path.join(ano_dir, "manuelle", f"{location}.csv")
        df_location = pd.read_csv(fname_location, index_col=0)
        df_location.index = pd.to_datetime(df_location.index)
        df_location_test = df_location[df_location.index.year == 2016]
        df_location_test = df_location_test.values.flatten()

        fname_location = os.path.join(processed_dir, f"{norm_location}.csv")
        df_norm_location = pd.read_csv(fname_location, index_col=0)
        df_norm_location.index = pd.to_datetime(df_norm_location.index)
        df_norm_location_test = df_norm_location[df_norm_location.index.year == 2016]
        df_norm_location_test = df_norm_location_test.values.flatten()

        fname_location = os.path.join(processed_dir, f"{location}.csv")
        df_orig_location = pd.read_csv(fname_location, index_col=0)
        df_orig_location.index = pd.to_datetime(df_orig_location.index)
        df_orig_location_test = df_orig_location[df_orig_location.index.year == 2016]
        df_orig_location_test = df_orig_location_test.values.flatten()

        count_original, bins_count_original = np.histogram(
            df_orig_location_test, bins=24
        )
        # finding the PDF of the histogram using count values
        pdf_original = count_original / sum(count_original)
        # using numpy np.cumsum to calculate the CDF
        # We can also find using the PDF values by looping and adding
        cdf_original = np.cumsum(pdf_original)

        # getting data of the histogram
        count_norm, bins_count_norm = np.histogram(df_norm_location_test, bins=24)
        # finding the PDF of the histogram using count values
        pdf_norm = count_norm / sum(count_norm)
        # using numpy np.cumsum to calculate the CDF
        # We can also find using the PDF values by looping and adding
        cdf_norm = np.cumsum(pdf_norm)

        # getting data of the histogram
        count_ano, bins_count_ano = np.histogram(df_location_test, bins=24)
        # finding the PDF of the histogram using count values
        pdf_ano = count_ano / sum(count_ano)
        # using numpy np.cumsum to calculate the CDF
        # We can also find using the PDF values by looping and adding
        cdf_ano = np.cumsum(pdf_ano)

        # mere numerical grid to plot densities (no statistical significance)
        x_ = np.linspace(0.0, 0.055, 1000)

        # estimate mean (mu) and standard deviation (sigma) for each column

        # set random state for reproducibility
        freqs, psd = signal.welch(df_orig_location_test)
        freqs_t, psd_t = signal.welch(df_norm_location_test)
        freqs_v, psd_v = signal.welch(df_location_test)

        plt.figure(figsize=(5, 4))
        plt.semilogx(freqs, psd)
        plt.semilogx(freqs_t, psd_t)
        plt.semilogx(freqs_v, psd_v)
        plt.title("PSD: power spectral density")
        plt.xlabel("Frequency")
        plt.ylabel("Power")
        plt.tight_layout()

        # plt.show()

        fname = os.path.join("pdf_cdf", f"{location}_pdf_cdf")
        check_make_dir(fname, True)
        # plotting PDF and CDF
        plt.plot(
            bins_count_original[1:], pdf_original, color="red", label="PDF Original"
        )
        plt.plot(bins_count_original[1:], cdf_original, label="CDF Original")

        plt.plot(bins_count_norm[1:], pdf_norm, color="blue", label="PDF Norm")
        plt.plot(bins_count_norm[1:], cdf_norm, color="green", label="CDF Norm")

        plt.plot(bins_count_ano[1:], pdf_ano, color="yellow", label="PDF Ano")
        plt.plot(bins_count_ano[1:], cdf_ano, color="orange", label="CDF Ano")

        plt.legend()
        plt.savefig(fname)
