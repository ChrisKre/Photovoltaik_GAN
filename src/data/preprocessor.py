import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stl.mstl import MSTL

from src.data.settings import locations, raw_dir, processed_dir, merged_dir
from src.utils import check_make_dir


class Preprocessor:
    def __init__(self):
        self.locations = locations
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir

    def merge_raw(self, usecols: [] = ["GlobInc"]):
        df = self.load_locations(usecols)
        for location in self.locations.keys():
            df_location = self.split_by_location(df, location, "GlobInc")
            save_dir = merged_dir
            check_make_dir(save_dir)
            save_fname = os.path.join(save_dir, location + ".csv")
            df_location.to_csv(save_fname)

    def make_dataset(
        self, scale: bool = True, season_decomp: bool = False, usecols: [] = ["GlobInc"]
    ):
        """
        Create the dataset for all locations, each locations data will be saved as csf file
        :param scale:
        :param season_decomp:
        :param usecols:
        :return:
        """
        # Load data and annotate label ('Location')
        df = self.load_locations(usecols)
        save_dir = processed_dir

        # Min max scale of feature columns
        if scale:
            df = self.min_max_scale(df)
            save_dir = os.path.join(save_dir, "scaled")
        if season_decomp:
            save_dir = os.path.join(save_dir, "season_decomposed")

        dataset = {}
        # Split the data by location, split is done here because scaling is done on whole data
        for location in self.locations.keys():
            df_location = self.split_by_location(df, location, "GlobInc")
            if season_decomp:
                df_location = self.season_decomp(df_location)
                # Group data so that, each instance of the dataset represents the daily data (24 datapoints)
                df_location = self.group_by_day_season_decomp(df_location)
            else:
                df_location = self.group_by_day(df_location, "GlobInc")
            check_make_dir(save_dir)
            save_fname = os.path.join(save_dir, location + ".csv")
            df_location.to_csv(save_fname)
        self.dataset = dataset

    def load_location(self, dir: str, dtype: {}, names: {}, usecols: [], decimal: str):
        """
        Load all csv files for a specific location found in given data dir and concat them together
        :param dir:
        :param dtype:
        :param names:
        :param usecols:
        :param decimal:
        :return:
        """
        df = pd.DataFrame()
        for file in os.listdir(dir):
            filename = os.path.join(dir, file)
            df_new = pd.read_csv(
                filename,
                sep=";",
                skiprows=12,
                dtype=dtype,
                names=names,
                encoding="latin1",
                usecols=usecols,
                decimal=decimal,
            )

            df = pd.concat([df, df_new], ignore_index=True)
        df.Datum = pd.to_datetime(df.Datum)
        df = df.sort_values(by=["Datum"])
        # Some locations have different times e.g. 08:10 instead of 08:00
        df["Datum"] = df.assign(Date=df["Datum"].dt.round("H"))["Date"]
        df = df.set_index("Datum")
        return df

    def load_locations(self, usecols: [] = ["GlobInc"]):
        """
        Load all csv files for a all locations found in given data dir and concat them together
        :param usecols:
        :return:
        """
        usecols.append("Datum")
        df = pd.DataFrame()
        for location, location_setting in self.locations.items():
            dir = os.path.join(raw_dir, location)
            df_location = self.load_location(
                dir,
                location_setting["dtype"],
                location_setting["names"],
                usecols,
                location_setting["decimal"],
            )
            df_location["Location"] = location
            df = pd.concat([df, df_location])
        return df

    def min_max_scale(self, df: pd.DataFrame):
        """
        MinMax scale data for a given dataframe
        :param df:
        :param column:
        :return:
        """

        self.scaler = MinMaxScaler()
        df[df.columns.difference(["Location"])] = self.scaler.fit_transform(
            df[df.columns.difference(["Location"])]
        )
        return df

    def group_by_day(self, df: pd.DataFrame, column: str = "GlobInc"):
        """
        Split the dataframe into lists of daily data,
        each row of the dataframe contains a list of 24 data points
        :param df:
        :return:
        """

        df = df.reset_index()
        df["Datum"] = df["Datum"].dt.date
        feature_list = df.groupby(["Datum"])[column].apply(list)
        feature_array = np.array([np.array(xi) for xi in feature_list.values])
        df_grouped = pd.DataFrame(feature_array, index=feature_list.index)
        return df_grouped

    def split_by_location(self, df: pd.DataFrame, location: str, column: []):
        """
        Extract all data for a given location
        :param df:
        :param location:
        :param column:
        :return:
        """
        df = df[df["Location"] == location]
        df = df.reset_index()
        feature = np.array(df[column].values.tolist())
        df_location = pd.DataFrame(feature, index=df["Datum"], columns=[column])
        return df_location

    def season_decomp(self, df_location: pd.DataFrame):
        """
        Run multi seasonal trend decomposition
        -  This can take some time
        :param df_location:
        :return:
        """
        print("START MSTL")
        res = MSTL(df_location["GlobInc"], periods=(24, 24 * 365)).fit()
        print("MSTL ENDED")
        df = pd.DataFrame()
        df["seasonal_24"] = res.seasonal["seasonal_24"]
        df["seasonal_8760"] = res.seasonal["seasonal_8760"]
        df["resid"] = res.resid
        df["trend"] = res.trend
        df["observed"] = res.observed

        return df

    def group_by_day_season_decomp(self, df_location):
        df_out = pd.DataFrame()
        for column in df_location.columns:
            df_tmp = self.group_by_day(df_location, column)
            df_tmp = pd.DataFrame({column: df_tmp.values.tolist()}, index=df_tmp.index)
            df_out[column] = df_tmp

        return df_out


if __name__ == "__main__":
    Preprocessor().make_dataset(season_decomp=False, scale=False)
