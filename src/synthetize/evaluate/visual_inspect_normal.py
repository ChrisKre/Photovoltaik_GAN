"""
Plotted sample von originalen und synthetischen Stichproben
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.synthetize.evaluate.utils import auto_correlation_normal

root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(os.path.join(root))
from src.data.settings import processed_dir, synth_dir

if __name__ == "__main__":
    scaled = True
    sd = False
    locations = ["Koethen", "Le_Havre", "Madrid", "München"]

    # Define dates for visual inspection
    dates = [
        pd.Timestamp(year=2016, month=3, day=12),
        pd.Timestamp(year=2016, month=6, day=22),
        pd.Timestamp(year=2016, month=7, day=1),
        pd.Timestamp(year=2016, month=2, day=3),
        pd.Timestamp(year=2016, month=9, day=8),
        pd.Timestamp(year=2016, month=12, day=4),
    ]

    # Create plots for each location
    for location in locations:
        # Load real data
        fname = os.path.join(processed_dir, "scaled", f"{location}.csv")
        df = pd.read_csv(fname, index_col=0)
        df.index = pd.to_datetime(df.index)

        # Load data from timegan
        fname_synth_timegan = os.listdir(
            os.path.join(synth_dir, "time_gan", "normal", location)
        )[0]
        fname_synth_timegan = os.path.join(
            synth_dir, "time_gan", "normal", location, fname_synth_timegan
        )
        df_synth_timegan = pd.read_csv(fname_synth_timegan, index_col=0)

        # Load data from vaegan
        fname_synth_vaegan = os.listdir(
            os.path.join(synth_dir, "vae_gan", "normal", location)
        )[0]
        fname_synth_vaegan = os.path.join(
            synth_dir, "vae_gan", "normal", location, fname_synth_vaegan
        )
        df_synth_vaegan = pd.read_csv(fname_synth_vaegan, index_col=0)

        # Get Sample with closest distance to original for timegan
        samples_timegan = []
        for date in dates:
            df_date = df[df.index == date]
            d = 50
            out = 0
            for i, synth_data in df_synth_timegan.iterrows():
                d_tmp = np.linalg.norm(synth_data - df_date)
                if d_tmp < d:
                    out = synth_data
                    d = d_tmp

            samples_timegan.append((out, df_date))

        # Get Sample with closest distance to original for vaegan
        samples_vaegan = []
        for date in dates:
            df_date = df[df.index == date]
            d = 50
            out = 0
            for i, synth_data in df_synth_vaegan.iterrows():
                d_tmp = np.linalg.norm(synth_data - df_date)
                if d_tmp < d:
                    out = synth_data
                    d = d_tmp
            samples_vaegan.append((out, df_date))

        # Plot all samples
        index = [i for i in range(0, 24)]
        fig, ax = plt.subplots(nrows=6, ncols=4)
        for i, (sample_timegan, sample_vaegan) in enumerate(
            zip(samples_timegan, samples_vaegan)
        ):
            synth_timegan_correlation = auto_correlation_normal(sample_timegan[0])
            synth_vaegan_correlation = auto_correlation_normal(sample_vaegan[0])
            real_correlation = auto_correlation_normal(sample_timegan[1].to_numpy()[0])

            ax[i, 0].plot(index, sample_timegan[1].to_numpy()[0])
            ax[i, 1].plot(index, sample_timegan[0])
            ax[i, 2].plot(index, sample_vaegan[0])
            ax[i, 3].plot(index, synth_timegan_correlation, color="blue")
            ax[i, 3].plot(index, synth_vaegan_correlation, color="green")
            ax[i, 3].plot(index, real_correlation, color="purple")

        plt.show()
