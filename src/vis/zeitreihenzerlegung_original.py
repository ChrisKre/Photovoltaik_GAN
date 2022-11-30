"""
Plotte zerlegte Daten der Standorte
"""

import os

from matplotlib import pyplot as plt

from src.data.preprocessor import Preprocessor
from src.data.settings import locations
from src.data.settings import processed_dir, merged_dir
from src.utils import plot_dir, check_make_dir

if __name__ == "__main__":
    location = "Koethen.csv"
    file_dir = processed_dir
    p = Preprocessor()
    df = p.load_locations()
    df = p.min_max_scale(df)

    for loc in locations:
        save_fname = os.path.join(plot_dir, loc, f"{loc}_seasond_decomp.png")
        file_dir_save = os.path.join(merged_dir, "season_decomp")
        save_fname_csv = os.path.join(file_dir_save, f"{loc}.csv")
        check_make_dir(save_fname_csv, True)
        df_location = p.split_by_location(df, loc, "GlobInc")

        df_location = p.season_decomp(df_location)
        df_location.to_csv(save_fname_csv)

        f, axes = plt.subplots(5, 1, figsize=(18, 24), dpi=200)

        axes[0].plot(df_location["observed"], color="green", label="Beobachtete Daten")
        axes[0].legend(loc="upper right", prop={"size": 16})
        # plotting daily seasonal component
        axes[1].plot(
            df_location["seasonal_24"].iloc[:1000],
            linewidth=2,
            label="Tägliche saisonale Komponente",
        )
        # plotting weekly seasonal component
        axes[1].legend(loc="upper right", prop={"size": 16})
        axes[2].plot(
            df_location["seasonal_8760"].iloc[:8760],
            color="orange",
            label="Jährliche saisonale Komponente",
        )
        axes[2].legend(loc="upper right", prop={"size": 16})
        # plotting trend component
        axes[3].plot(df_location["trend"], color="r", label="Trend Komponente")
        axes[3].legend(loc="upper right", prop={"size": 16})
        # plotting yearly seasonality
        axes[4].plot(df_location["resid"], color="green", label="Rauschkomponente")
        axes[4].legend(loc="upper right", prop={"size": 16})
        # plotting residual of decomposition
        for a in axes:
            a.set_ylabel("W/m²", fontsize=14)
            a.tick_params(axis="y", labelsize=12)
            a.tick_params(axis="x", labelsize=12)
        plt.tight_layout()
        check_make_dir(save_fname, True)
        plt.savefig(save_fname)
