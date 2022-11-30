import os
from ast import literal_eval

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.settings import processed_dir, classification_labels, synth_dir
from src.utils import seasonal_decomposed_cols


class DataLoader:
    def list_to_array(self, df: pd.DataFrame):
        """
        Convert list format in dataframe rows into numpy array
        :param df:
        :return:
        """
        out = []
        for i, row in df.iterrows():
            #
            # tmp = np.array([np.array(row['resid_daily']), np.array(row['trend_daily']), np.array(row['seasonal_daily'])])
            # # s = np.expand_dims(tmp, axis=0)
            # tmp = np.transpose(tmp, (1, 0))
            tmp = np.transpose(row.tolist(), (1, 0))
            out.append(tmp)
        out = np.array(out)
        return out

    def get_dataset_shapes(self):
        """
        Return shapes of dataset
        :return:
        """
        input_shape = tuple(self.training_dataset.element_spec[0].shape)
        num_classes = (
            2
            if self.training_dataset.element_spec[1].shape[0] == 1
            else self.training_dataset.element_spec[1].shape[0]
        )
        return input_shape, num_classes

    def get_train_test_data(self):
        return self.training_dataset, self.test_dataset

    def get_dataset(self, x: str, scaled: bool = True, season_decomp: bool = False):
        file_dir = processed_dir
        x_fname = os.path.join(file_dir, x + ".csv")
        df_x = pd.read_csv(x_fname, index_col=0)
        df_x.index = pd.to_datetime(df_x.index)

    def load_to_dataframe(
        self, filename: str, season_decomp: bool = True, test_year: int = None
    ):
        if season_decomp:
            df = pd.read_csv(
                filename,
                index_col=0,
                converters={
                    seasonal_decomposed_cols[0]: literal_eval,
                    seasonal_decomposed_cols[1]: literal_eval,
                    seasonal_decomposed_cols[2]: literal_eval,
                    seasonal_decomposed_cols[3]: literal_eval,
                },
            )
            df = df[seasonal_decomposed_cols]

        else:
            df = pd.read_csv(filename, index_col=0)
        df.index = pd.to_datetime(df.index)
        if test_year is not None:
            return df[df.index.year == test_year]
        return df

    def load_to_numpy(self, filename: str, season_decomp: bool = True):
        if season_decomp == True:
            return self.load_season_decomp(filename)

        else:
            df = pd.read_csv(filename, index_col=0)
            return np.expand_dims(df.values, axis=2)

    def load_season_decomp(self, filename: str):
        df = pd.read_csv(
            filename,
            index_col=0,
            converters={
                seasonal_decomposed_cols[0]: literal_eval,
                seasonal_decomposed_cols[1]: literal_eval,
                seasonal_decomposed_cols[2]: literal_eval,
                seasonal_decomposed_cols[3]: literal_eval,
            },
        )
        df = df[seasonal_decomposed_cols]
        synth = self.list_to_array(df)
        return synth

    def get_by_season(self, x: np.array, season: int = 0):
        if season == 0:
            index = np.where(
                (x.index.month == 3) | (x.index.month == 4) | (x.index.month == 5)
            )
        elif season == 1:
            index = np.where(
                (x.index.month == 6) | (x.index.month == 7) | (x.index.month == 8)
            )
        elif season == 2:
            index = np.where(
                (x.index.month == 9) | (x.index.month == 10) | (x.index.month == 11)
            )
        elif season == 3:
            index = np.where(
                (x.index.month == 12) | (x.index.month == 1) | (x.index.month == 2)
            )

        return x.iloc[index]


class DataLoaderEnc(DataLoader):
    def __init__(
        self,
        x: str,
        y: str,
        scaled: bool = True,
        season_decomp: bool = False,
        test_year: int = 2016,
    ):
        self.training_dataset, self.test_dataset = self.get_dataset(
            x, y, scaled, season_decomp, test_year
        )
        self.test_year = test_year

    def get_dataset(
        self,
        x: str,
        y: str,
        scaled: bool = True,
        season_decomp: bool = False,
        test_year: int = 2016,
    ):
        """
        Creates tf.data.Dataset for given configuration
        :param x: input location for training
        :param y: target location for training
        :param scaled: whether the data is minmax scaled
        :param season_decomp: whether the data is seasonal decomposed
        :param test_year: will be used for test dataset
        :return:
        """
        file_dir = processed_dir
        if scaled:
            file_dir = os.path.join(file_dir, "scaled")
        if season_decomp:
            training_dataset, test_dataset = self.get_dataset_season_decomp(
                file_dir, x, y, test_year
            )
        else:
            x_fname = os.path.join(file_dir, x + ".csv")
            y_fname = os.path.join(file_dir, y + ".csv")
            df_x = pd.read_csv(x_fname, index_col=0)
            df_y = pd.read_csv(y_fname, index_col=0)

            df_x.index = pd.to_datetime(df_x.index)
            df_y.index = pd.to_datetime(df_y.index)

            df_x_train = df_x[df_x.index.year != test_year]
            df_x_test = df_x[df_x.index.year == test_year]
            df_y_train = df_y[df_y.index.year != test_year]
            df_y_test = df_y[df_y.index.year == test_year]
            training_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    tf.cast(np.expand_dims(df_x_train.values, axis=2), tf.float32),
                    tf.cast(np.expand_dims(df_y_train.values, axis=2), tf.float32),
                )
            )

            test_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    tf.cast(np.expand_dims(df_x_test.values, axis=2), tf.float32),
                    tf.cast(np.expand_dims(df_y_test.values, axis=2), tf.float32),
                )
            )
        return training_dataset, test_dataset

    def get_dataset_season_decomp(self, file_dir: str, x: str, y: str, test_year: int):
        """
        Create dataset for seasonal decomposed data
        :param file_dir:
        :param x:
        :param y:
        :param test_year:
        :return:
        """
        file_dir = os.path.join(file_dir, "season_decomposed")
        x_fname = os.path.join(file_dir, x + ".csv")
        y_fname = os.path.join(file_dir, y + ".csv")
        df_x = pd.read_csv(
            x_fname,
            index_col=0,
            converters={
                "resid": literal_eval,
                "trend": literal_eval,
                "seasonal": literal_eval,
            },
        )
        df_y = pd.read_csv(
            y_fname,
            index_col=0,
            converters={
                "resid": literal_eval,
                "trend": literal_eval,
                "seasonal": literal_eval,
            },
        )
        df_x.index = pd.to_datetime(df_x.index)
        df_y.index = pd.to_datetime(df_y.index)
        df_x_train = df_x[df_x.index.year != test_year]
        df_x_test = df_x[df_x.index.year == test_year]
        df_y_train = df_y[df_y.index.year != test_year]
        df_y_test = df_y[df_y.index.year == test_year]
        x_train = self.list_to_array(df_x_train)
        x_test = self.list_to_array(df_x_test)
        y_train = self.list_to_array(df_y_train)
        y_test = self.list_to_array(df_y_test)
        x_train = tf.data.Dataset.from_tensor_slices(
            (tf.cast(x_train, tf.float32), tf.cast(y_train, tf.float32))
        )
        y_test = tf.data.Dataset.from_tensor_slices(
            (tf.cast(x_test, tf.float32), tf.cast(y_test, tf.float32))
        )
        return x_train, y_test

    def get_dataset_shapes(self):
        input_shape = tuple(self.training_dataset.element_spec[0].shape)
        return input_shape


class DataLoaderEncSing(DataLoader):
    def __init__(
        self,
        x: str,
        scaled: bool = True,
        season_decomp: bool = False,
        test_year: int = 2016,
        season: int = None,
    ):
        self.test_year = test_year
        self.season_decomp = season_decomp
        self.season = season
        self.training_dataset, self.test_dataset = self.get_dataset(
            x, scaled, season_decomp, test_year
        )

    def get_dataset(
        self,
        x: str,
        scaled: bool = True,
        season_decomp: bool = False,
        test_year: int = 2016,
    ):
        """
        Creates tf.data.Dataset for given configuration
        :param x: input location for training
        :param scaled: whether the data is minmax scaled
        :param season_decomp: whether the data is seasonal decomposed
        :param test_year: will be used for test dataset
        :return:
        """
        file_dir = processed_dir
        if scaled:
            file_dir = os.path.join(file_dir, "scaled")
        if season_decomp:
            training_dataset, test_dataset = self.get_dataset_season_decomp(
                file_dir, x, test_year
            )
        else:
            x_fname = os.path.join(file_dir, x + ".csv")
            df_x = pd.read_csv(x_fname, index_col=0)

            df_x.index = pd.to_datetime(df_x.index)
            if self.season is not None:
                df_x = self.get_by_season(df_x, self.season)
            df_x_train = df_x[df_x.index.year != test_year]
            df_x_test = df_x[df_x.index.year == test_year]
            training_dataset = tf.data.Dataset.from_tensor_slices(
                (tf.cast(np.expand_dims(df_x_train.values, axis=2), tf.float32),)
            )

            test_dataset = tf.data.Dataset.from_tensor_slices(
                (tf.cast(np.expand_dims(df_x_test.values, axis=2), tf.float32),)
            )
        return training_dataset, test_dataset

    def get_dataset_season_decomp(self, file_dir: str, x: str, test_year: int):
        """
        Create dataset for seasonal decomposed data
        :param file_dir:
        :param x:
        :param y:
        :param test_year:
        :return:
        """
        file_dir = os.path.join(file_dir, "season_decomposed")
        x_fname = os.path.join(file_dir, x + ".csv")

        df = pd.read_csv(
            x_fname,
            index_col=0,
            converters={
                seasonal_decomposed_cols[0]: literal_eval,
                seasonal_decomposed_cols[1]: literal_eval,
                seasonal_decomposed_cols[2]: literal_eval,
                seasonal_decomposed_cols[3]: literal_eval,
            },
        )
        df_x = df[seasonal_decomposed_cols]
        df_x.index = pd.to_datetime(df_x.index)
        df_x_train = df_x[df_x.index.year != test_year]
        df_x_test = df_x[df_x.index.year == test_year]
        x_train = self.list_to_array(df_x_train)
        x_test = self.list_to_array(df_x_test)
        x_train = tf.data.Dataset.from_tensor_slices((tf.cast(x_train, tf.float32)))
        x_test = tf.data.Dataset.from_tensor_slices((tf.cast(x_test, tf.float32)))
        return x_train, x_test

    def get_dataset_shapes(self):
        if self.season_decomp:
            input_shape = tuple(self.training_dataset.element_spec.shape)
        else:
            input_shape = tuple(self.training_dataset.element_spec[0].shape)
        return input_shape


class DataLoaderCl(DataLoader):
    def __init__(
        self,
        x: [str] = ["Madrid", "Koethen"],
        scaled: bool = True,
        season_decomp: bool = False,
        test_year: int = 2016,
        binary: bool = False,
    ):
        self.binary = binary
        self.training_dataset, self.test_dataset = self.get_dataset(
            x, scaled, season_decomp, test_year
        )

    def get_dataset(
        self,
        x: str,
        scaled: bool = True,
        season_decomp: bool = False,
        test_year: int = 2016,
    ):
        file_dir = processed_dir
        if scaled:
            file_dir = os.path.join(file_dir, "scaled")
        if season_decomp:
            training_dataset, test_dataset = self.get_dataset_season_decomp(
                file_dir, x, test_year
            )
        else:
            df_train, df_test = pd.DataFrame(), pd.DataFrame()
            for i, x_location in enumerate(x):
                location_file_name = os.path.join(file_dir, x_location + ".csv")
                df_train_location, df_test_location = self.get_location_data(
                    location_file_name, x_location, test_year, i
                )
                df_train, df_test = pd.concat(
                    [df_train, df_train_location], ignore_index=True
                ), pd.concat([df_test, df_test_location], ignore_index=True)

            if len(df_train["target"].unique()) > 2:
                train_target = tf.keras.utils.to_categorical(
                    df_train["target"], num_classes=len(df_train["target"].unique())
                )
                test_target = tf.keras.utils.to_categorical(
                    df_test["target"], num_classes=len(df_test["target"].unique())
                )
            else:
                train_target = tf.cast(
                    np.asarray(df_train["target"].values).reshape((-1, 1)), tf.int16
                )
                test_target = tf.cast(
                    np.asarray(df_test["target"].values).reshape((-1, 1)), tf.int16
                )

            training_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    tf.cast(
                        np.expand_dims(
                            df_train[df_train.columns.difference(["target"])].values,
                            axis=2,
                        ),
                        tf.float32,
                    ),
                    train_target,
                )
            )

            test_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    tf.cast(
                        np.expand_dims(
                            df_test[df_test.columns.difference(["target"])].values,
                            axis=2,
                        ),
                        tf.float32,
                    ),
                    test_target,
                )
            )
        return training_dataset, test_dataset

    def get_location_data(self, file_name: str, location: str, test_year: int, i: int):
        df = pd.read_csv(file_name, index_col=0)
        df.index = pd.to_datetime(df.index)
        df_train = df[df.index.year != test_year]
        df_test = df[df.index.year == test_year]
        if self.binary:
            df_train["target"] = i
            df_test["target"] = i
        else:
            df_train["target"] = classification_labels[location]
            df_test["target"] = classification_labels[location]
        return df_train, df_test

    def get_dataset_season_decomp(
        self, file_dir: str, x: [str] = ["Madrid", "Koethen"], test_year: int = 2016
    ):
        train_x, train_y, test_x, test_x = None, None, None, None
        for i, x_location in enumerate(x):
            location_file_name = os.path.join(
                file_dir, "season_decomposed", x_location + ".csv"
            )
            df_location = pd.read_csv(
                location_file_name,
                index_col=0,
                converters={
                    "seasonal_24": literal_eval,
                    "seasonal_8760": literal_eval,
                    "resid": literal_eval,
                    "trend": literal_eval,
                },
            )
            df_location = df_location[seasonal_decomposed_cols]

            df_location.index = pd.to_datetime(df_location.index)
            df_location_train = df_location[df_location.index.year != test_year]
            df_location_test = df_location[df_location.index.year == test_year]
            x_train = self.list_to_array(df_location_train)
            x_test = self.list_to_array(df_location_test)
            if self.binary:
                y_train = np.full(fill_value=i, shape=x_train.shape[0])
                y_test = np.full(fill_value=i, shape=x_test.shape[0])
            else:
                y_train = np.full(
                    fill_value=classification_labels[x_location], shape=x_train.shape[0]
                )
                y_test = np.full(
                    fill_value=classification_labels[x_location], shape=x_test.shape[0]
                )
            if i == 0:
                train_x, train_y, test_x, test_y = x_train, y_train, x_test, y_test
            else:
                train_x = np.vstack((train_x, x_train))
                train_y = np.hstack((train_y, y_train))
                test_x = np.vstack((test_x, x_test))
                test_y = np.hstack((test_y, y_test))

        if len(np.unique(train_y)) > 2:
            train_target = tf.keras.utils.to_categorical(
                train_y, num_classes=len(np.unique(train_y))
            )
            test_target = tf.keras.utils.to_categorical(
                test_y, num_classes=len(np.unique(test_y))
            )
        else:
            train_target = tf.cast(train_y.reshape((-1, 1)), tf.int16)
            test_target = tf.cast(test_y.reshape((-1, 1)), tf.int16)

        training_dataset = tf.data.Dataset.from_tensor_slices(
            (tf.cast(train_x, tf.float32), train_target)
        )

        test_dataset = tf.data.Dataset.from_tensor_slices(
            (tf.cast(test_x, tf.float32), test_target)
        )
        return training_dataset, test_dataset


class DataLoaderSynth(DataLoader):
    def __init__(
        self,
        x: [str] = ["Madrid", "Koethen"],
        scaled: bool = True,
        season_decomp: bool = True,
        test_year: int = 2016,
        time_gan: bool = True,
    ):
        self.training_dataset = self.get_dataset(
            x, scaled, season_decomp, test_year, time_gan
        )

    def get_dataset(
        self,
        x: [str],
        scaled: bool = True,
        season_decomp: bool = False,
        test_year: int = 2016,
        lstm: bool = True,
        time_gan: bool = True,
    ):
        """
        Creates tf.data.Dataset for given configuration
        :param x: input location for training
        :param scaled: whether the data is minmax scaled
        :param season_decomp: whether the data is seasonal decomposed
        :param test_year: will be used for test dataset
        :return:
        """
        file_dir = synth_dir
        if time_gan:
            file_dir = os.path.join(file_dir, "time_gan")
        else:
            file_dir = os.path.join(file_dir, "vae_gan")

        if season_decomp:
            training_dataset = self.get_dataset_season_decomp(file_dir, x)
        else:
            df_train = pd.DataFrame()
            for i, x_location in enumerate(x):
                location_file_dir = os.path.join(file_dir, "normal", x_location)

                df_train_location = self.get_location_data(location_file_dir, i)
                df_train = pd.concat([df_train, df_train_location], ignore_index=True)

            if len(df_train["target"].unique()) > 2:
                train_target = tf.keras.utils.to_categorical(
                    df_train["target"], num_classes=len(df_train["target"].unique())
                )
            else:
                train_target = tf.cast(
                    np.asarray(df_train["target"].values).reshape((-1, 1)), tf.int16
                )

            training_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    tf.cast(
                        np.expand_dims(
                            df_train[df_train.columns.difference(["target"])].values,
                            axis=2,
                        ),
                        tf.float32,
                    ),
                    train_target,
                )
            )

        return training_dataset

    def get_training_data(self):
        return self.training_dataset

    def get_dataset_season_decomp(self, file_dir: str, x: str):
        """
        Create dataset for seasonal decomposed data
        :param file_dir:
        :param x:
        :param y:
        :param test_year:
        :return:
        """
        for i, x_location in enumerate(x):
            file_dir_x = os.path.join(file_dir, "season_decomposed")
            file_dir_x = os.path.join(file_dir_x, x_location)
            x_fname = os.path.join(file_dir_x, os.listdir(file_dir_x)[0])

            df_x = pd.read_csv(x_fname, index_col=0)
            for col in df_x.columns:
                df_x[col] = df_x[col].apply(literal_eval)
            df_x.index = pd.to_datetime(df_x.index)
            x_train = self.list_to_array(df_x)

            y_train = np.full(
                fill_value=classification_labels[x_location], shape=x_train.shape[0]
            )
            if i == 0:
                train_x, train_y = x_train, y_train
            else:
                train_x = np.vstack((train_x, x_train))
                train_y = np.hstack((train_y, y_train))

        if len(np.unique(train_y)) > 2:
            train_target = tf.keras.utils.to_categorical(
                train_y, num_classes=len(np.unique(train_y))
            )
        else:
            train_target = tf.cast(train_y.reshape((-1, 1)), tf.int16)

        training_dataset = tf.data.Dataset.from_tensor_slices(
            (tf.cast(train_x, tf.float32), train_target)
        )

        return training_dataset

    def get_dataset_shapes(self):
        return (
            self.training_dataset.element_spec[0].shape,
            self.training_dataset.element_spec[1].shape[0],
        )

    def get_train_data(self):
        return self.training_dataset

    def get_location_data(self, file_dir, i):
        x_fname = os.listdir(file_dir)[0]
        x_fname = os.path.join(file_dir, x_fname)
        df_x = pd.read_csv(x_fname, index_col=0)
        df_x["target"] = i
        return df_x
