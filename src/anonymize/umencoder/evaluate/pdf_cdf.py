import argparse
import os.path

from src.anonymize.vae_gan_ano.evaluate.pdf_cdf import plot_pdf_cdf, plot_pds
from src.data.dataloader import DataLoader
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
    dl = DataLoader()
    test_year = 2016
    for location in locations:
        fname_location = os.path.join(ano_dir, "umencoder", f"{location}.csv")
        df_location = dl.load_to_dataframe(fname_location, season_decomp=False)
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
