import os

import pandas as pd
from matplotlib import pyplot as plt

from src.data.settings import processed_dir, ano_dir


def to_string(norm, original, ano):
    norm_string = "Normierung: \t"
    original_string = "Original: \t"
    ano_string = "Anonymisierung: \t"
    for i, (n, o, a) in enumerate(zip(norm, original, ano)):
        norm_string += f"({i},{n})"
        original_string += f"({i},{o})"
        ano_string += f"({i},{a})"

    return f"{original_string}\n  {norm_string} \n {ano_string}\n"


if __name__ == "__main__":
    location = "Madrid"
    norm_location = "Koethen"
    test_year = 2016

    fname_location = os.path.join(processed_dir, "scaled", f"{norm_location}.csv")
    df_norm_location = pd.read_csv(fname_location, index_col=0)
    df_norm_location.index = pd.to_datetime(df_norm_location.index)

    fname_location = os.path.join(processed_dir, "scaled", f"{location}.csv")
    df_orig_location = pd.read_csv(fname_location, index_col=0)
    df_orig_location.index = pd.to_datetime(df_orig_location.index)

    fname_ano = os.path.join(ano_dir, "umencoder", f"{location}.csv")
    df_ano_location = pd.read_csv(fname_ano, index_col=0)
    df_ano_location.index = pd.to_datetime(df_ano_location.index)

    dates = [
        pd.Timestamp(year=2016, month=1, day=23),
        pd.Timestamp(year=2016, month=12, day=27),
        pd.Timestamp(year=2016, month=3, day=12),
        pd.Timestamp(year=2016, month=4, day=27),
        pd.Timestamp(year=2016, month=6, day=28),
        pd.Timestamp(year=2016, month=7, day=7),
        pd.Timestamp(year=2016, month=9, day=14),
        pd.Timestamp(year=2016, month=11, day=20),
    ]

    fig, ax = plt.subplots(nrows=len(dates), ncols=1)
    plot_index = [i for i in range(0, 24)]
    plot_string = ""
    for i, date in enumerate(dates):
        norm_example = df_norm_location[df_norm_location.index == date].iloc[0]
        original_example = df_orig_location[df_orig_location.index == date].iloc[0]
        ano_example = df_ano_location[df_ano_location.index == date].iloc[0]
        ax[i].plot(plot_index, norm_example, label="Normierung")
        ax[i].plot(plot_index, original_example, label="Original")
        ax[i].plot(plot_index, ano_example, label="Anonymisierung")
        plot_string += (
            f"{str(date)}\n {to_string(norm_example, original_example, ano_example)} \n"
        )

    plt.legend()
    plt.show()

    plot_txt = os.path.join(location, "examples.txt")
    # open text file
    text_file = open(plot_txt, "w")
    # write string to file
    text_file.write(plot_string)
    # close file
    text_file.close()
