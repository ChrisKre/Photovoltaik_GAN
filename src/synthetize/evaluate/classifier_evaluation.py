"""
Helferklasse für die Evaluierung der synthetischen Datensätze
"""

import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.classificator.classificator import Classificator
from src.data.dataloader import DataLoaderCl, DataLoaderSynth
from src.utils import get_model_dir
from src.synthetize.evaluate.utils import classifier_predict


class ClassifierEvaluation:
    def __init__(self, file_dir: str, scaled: bool = True, season_decomp: bool = False):
        self.file_dir = file_dir
        self.scaled = scaled
        self.season_decomp = season_decomp
        pass

    def discriminative_score(self, real: np.ndarray, fake: np.ndarray):
        """
        Make real/fake classification for given data
        :param real:
        :param fake:
        :return:
        """
        real_target = np.full(shape=real.shape[0], fill_value=1)
        fake_target = np.full(shape=fake.shape[0], fill_value=0)

        x = np.concatenate((real, fake))
        y = np.concatenate((real_target, fake_target))
        timesteps, n_features = real.shape[1], real.shape[2]

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

        training_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))

        test_dataset = tf.data.Dataset.from_tensor_slices(
            (tf.cast(X_test, tf.float32), y_test)
        )

        cl = Classificator(
            input_dim=timesteps,
            num_feat=n_features,
            num_classes=2,
            eval_dir=self.file_dir,
            season_decomp=self.season_decomp,
        )

        cl.train(training_dataset, epochs=30)
        cl.eval(test_dataset, ["real", "synth"])

    def train_synth_test_real(self, time_gan: bool = True, test_year=2016):
        """
        Train Synthetic Test Real Evaluation for all locations
        :param time_gan:
        :param test_year:
        :return:
        """
        x = ["Madrid", "Koethen", "München", "Le_Havre"]

        dl = DataLoaderCl(x, self.scaled, self.season_decomp, test_year)
        _, test_dataset = dl.get_train_test_data()
        input_shape, num_classes = dl.get_dataset_shapes()

        dl = DataLoaderSynth(x, self.scaled, self.season_decomp, time_gan=time_gan)
        training_dataset = dl.get_training_data()

        cl = Classificator(
            input_dim=input_shape[0],
            num_feat=input_shape[1],
            num_classes=4,
            eval_dir=self.file_dir,
            season_decomp=self.season_decomp,
        )
        cl.train(training_dataset, epochs=100)
        cl.eval(test_dataset, x)

    def train_real_test_synth(self, synth: np.ndarray, location):
        model_dir = get_model_dir(sd=self.season_decomp)
        cl_model = tf.keras.models.load_model(os.path.join(model_dir, "model"))

        classifier_predict(cl_model, synth, location, self.file_dir)
