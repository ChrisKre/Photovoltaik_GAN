"""
Plotted sample von originalen und synthetischen Stichproben der zerlegten Daten
"""

import os
import sys
from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(os.path.join(root))
from src.data.dataloader import DataLoader
from src.utils import seasonal_decomposed_cols
from src.synthetize.evaluate.utils import auto_correlation
from src.data.settings import processed_dir, synth_dir

if __name__ == "__main__":
    scaled = True
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

    dl = DataLoader()
    # Create plots for each location
    for location in locations:
        fname = os.path.join(
            processed_dir, "scaled", "season_decomposed", f"{location}.csv"
        )
        df = pd.read_csv(
            fname,
            index_col=0,
            converters={
                seasonal_decomposed_cols[0]: literal_eval,
                seasonal_decomposed_cols[1]: literal_eval,
                seasonal_decomposed_cols[2]: literal_eval,
                seasonal_decomposed_cols[3]: literal_eval,
            },
        )[seasonal_decomposed_cols]
        df.index = pd.to_datetime(df.index)

        # Load data from timegan
        dir_synth_timegan = os.path.join(
            synth_dir, "time_gan", "season_decomposed", location
        )
        fname_synth_timegan = os.listdir(dir_synth_timegan)[0]
        fname_synth_timegan = os.path.join(dir_synth_timegan, fname_synth_timegan)
        df_synth_timegan = dl.load_to_dataframe(fname_synth_timegan, True)

        # Load data from vaegan
        dir_synth_vaegan = os.path.join(
            synth_dir, "vae_gan", "season_decomposed", "seasonal", location
        )
        fname_synth_vaegan = os.listdir(dir_synth_vaegan)[0]
        fname_synth_vaegan = os.path.join(dir_synth_vaegan, fname_synth_vaegan)
        df_synth_vaegan = dl.load_to_dataframe(fname_synth_vaegan, True)

        # Get Sample with closest distance to original for timegan
        samples_timegan = []
        for date in dates:
            df_date = df[df.index == date]
            d = 50
            out = 0
            # np.concatenate(synth_data.values.reshape(1, -1), axis=0)
            for i, synth_data in df_synth_timegan.iterrows():
                d_tmp = np.linalg.norm(
                    np.concatenate(synth_data.values, axis=0)
                    - np.concatenate(df_date.values[0], axis=0)
                )
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
                d_tmp = np.linalg.norm(
                    np.concatenate(synth_data.values, axis=0)
                    - np.concatenate(df_date.values[0], axis=0)
                )
                if d_tmp < d:
                    out = synth_data
                    d = d_tmp
            samples_vaegan.append((out, df_date))

        index = [i for i in range(0, 24)]

        # Plot all samples
        for i, (sample_timegan, sample_vaegan) in enumerate(
            zip(samples_timegan, samples_vaegan)
        ):
            fig, ax = plt.subplots(nrows=4, ncols=4)

            synth_timegan_correlation = auto_correlation(sample_timegan[0])
            synth_vaegan_correlation = auto_correlation(sample_vaegan[0])
            real_correlation = auto_correlation(sample_timegan[1].to_numpy()[0])

            ax[0, 0].plot(index, sample_timegan[1].to_numpy()[0][0])
            ax[0, 1].plot(index, sample_timegan[1].to_numpy()[0][1])
            ax[0, 2].plot(index, sample_timegan[1].to_numpy()[0][2])
            ax[0, 3].plot(index, sample_timegan[1].to_numpy()[0][3])

            ax[1, 0].plot(index, sample_timegan[0].to_numpy()[0])
            ax[1, 1].plot(index, sample_timegan[0].to_numpy()[1])
            ax[1, 2].plot(index, sample_timegan[0].to_numpy()[2])
            ax[1, 3].plot(index, sample_timegan[0].to_numpy()[3])

            ax[2, 0].plot(index, sample_vaegan[0].to_numpy()[0])
            ax[2, 1].plot(index, sample_vaegan[0].to_numpy()[1])
            ax[2, 2].plot(index, sample_vaegan[0].to_numpy()[2])
            ax[2, 3].plot(index, sample_vaegan[0].to_numpy()[3])

            ax[3, 0].plot(index, real_correlation[0], color="blue")
            ax[3, 0].plot(index, synth_timegan_correlation[0], color="green")
            ax[3, 0].plot(index, synth_vaegan_correlation[0], color="purple")

            ax[3, 1].plot(index, real_correlation[1], color="blue")
            ax[3, 1].plot(index, synth_timegan_correlation[1], color="green")
            ax[3, 1].plot(index, synth_vaegan_correlation[1], color="purple")

            ax[3, 2].plot(index, real_correlation[2], color="blue")
            ax[3, 2].plot(index, synth_timegan_correlation[2], color="green")
            ax[3, 2].plot(index, synth_vaegan_correlation[2], color="purple")

            ax[3, 3].plot(index, real_correlation[3], color="blue")
            ax[3, 3].plot(index, synth_timegan_correlation[3], color="green")
            ax[3, 3].plot(index, synth_vaegan_correlation[3], color="purple")

            plt.show()
            plt.close()
