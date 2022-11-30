"""
Erstelle VAE-GAN Modell
"""
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Activation
from tensorflow import keras

from src.data.settings import classification_labels
from src.utils import check_make_dir, get_model_dir

tf.random.set_seed(1234)


def sampling(args):
    """
    Sampling operation in VAE reparameter trick
    :param args:
    :return:
    """
    mean, logsigma = args
    epsilon = keras.backend.random_normal(shape=keras.backend.shape(mean))
    return mean + tf.exp(logsigma / 2) * epsilon


def encoder(timesteps: int = 24, n_features: int = 4, latent_depth: int = 3):
    """
    Return Encoder Model
    :param timesteps:
    :param n_features:
    :param latent_depth:
    :return:
    """
    input_E = keras.layers.Input(shape=(timesteps, n_features), name="input")

    X = keras.layers.Conv1D(
        filters=timesteps, kernel_size=2, strides=2, padding="same"
    )(input_E)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.ReLU()(X)

    X = keras.layers.Conv1D(
        filters=timesteps * 2, kernel_size=2, strides=2, padding="same"
    )(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.ReLU()(X)

    X = keras.layers.Conv1D(
        filters=timesteps * 4, kernel_size=2, strides=2, padding="same"
    )(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.ReLU()(X)

    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(2 * latent_depth)(X)

    mean = keras.layers.Dense(latent_depth, activation="tanh")(X)
    logsigma = keras.layers.Dense(latent_depth, activation="tanh")(X)
    latent = keras.layers.Lambda(sampling, output_shape=(latent_depth, n_features))(
        [mean, logsigma]
    )
    kl_loss = 1 + logsigma - keras.backend.square(mean) - keras.backend.exp(logsigma)
    kl_loss = keras.backend.mean(kl_loss, axis=-1)
    kl_loss *= -0.5

    return keras.models.Model(input_E, [latent, kl_loss])


def decoder(timesteps: int = 24, n_features: int = 4, latent_depth: int = 3):
    """
    Return Decoder Model
    :param timesteps:
    :param n_features:
    :param latent_depth:
    :return:
    """
    X = keras.layers.Input(shape=(3,))

    model = keras.layers.Dense(latent_depth * 4 * timesteps)(X)
    model = keras.layers.Reshape((latent_depth, 4 * timesteps))(model)
    model = keras.layers.BatchNormalization()(model)
    model = keras.layers.ReLU()(model)

    model = keras.layers.Conv1DTranspose(
        filters=timesteps * 2, kernel_size=2, strides=2, padding="same"
    )(model)
    model = keras.layers.BatchNormalization()(model)
    model = keras.layers.ReLU()(model)

    model = keras.layers.Conv1DTranspose(
        filters=timesteps, kernel_size=2, strides=2, padding="same"
    )(model)
    model = keras.layers.BatchNormalization()(model)
    model = keras.layers.ReLU()(model)

    model = keras.layers.Conv1DTranspose(
        filters=n_features, kernel_size=2, strides=2, padding="same"
    )(model)
    model = Activation("tanh")(model)

    model = keras.models.Model(X, model)
    return model


def discriminator(timesteps: int = 24, n_features: int = 1):
    """
    Return discriminator model
    :param timesteps: length of the sequenz -> 24
    :param n_features: number of features: 1 for normal data , 4 for season decomposed data
    :return:
    """
    input_D = keras.layers.Input(shape=(timesteps, n_features), name="input")

    X = keras.layers.Conv1D(
        filters=timesteps * 2, kernel_size=2, strides=2, padding="same"
    )(input_D)
    X = keras.layers.ReLU()(X)

    X = keras.layers.Conv1D(
        filters=timesteps * 4, kernel_size=2, strides=2, padding="same"
    )(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.ReLU()(X)

    X = keras.layers.Conv1D(
        filters=timesteps * 8, kernel_size=2, strides=2, padding="same"
    )(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.ReLU()(X)

    X = keras.layers.Conv1D(filters=timesteps * 8, kernel_size=2, padding="same")(X)
    inner_output = keras.layers.Flatten()(X)

    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.ReLU()(X)

    X = keras.layers.Flatten()(X)
    X = keras.layers.ReLU()(X)

    output = keras.layers.Dense(1)(X)

    return keras.models.Model(input_D, [output, inner_output])


class VAEGan:
    def __init__(
        self,
        timesteps: int = 24,
        n_features: int = 1,
        latent_depth: int = 12,
        batch_size: int = 32,
        weights: str = None,
        weights_dir: str = None,
        location: str = "Koethen",
        season_decomp: bool = False,
    ):
        """

        :param timesteps: Length of the sequenz -> 24
        :param n_features: Number of features: 1 for normal data , 4 for season decomposed data
        :param latent_depth: Dimensions of latent space for dimensionreduction
        :param batch_size: Size of batches during training
        :param weights: Needed to load a pretrained VAE-GAN model -> defines the number of iteration to load
        :param weights_dir: Needed to load a pretrained VAE-GAN model -> defines the direction where the weights are saved
        :param location: Name of the location to be synthetized
        :param season_decomp: Whether data is seasonal decomposed (True) or normal (False)
        """

        # Set Parameter
        self.season_decomp = season_decomp
        self.save_dir = get_model_dir("vae_gan", season_decomp)
        self.save_dir = os.path.join(self.save_dir, location)
        check_make_dir(self.save_dir)
        self.logging_dir = os.path.join(self.save_dir, "training_log")
        self.train_summary_writer = tf.summary.create_file_writer(self.logging_dir)
        self.timesteps = timesteps
        self.n_features = n_features
        self.latent_depth = latent_depth
        self.batch_size = batch_size
        self.location = location

        # Get Models
        self.enc = encoder(timesteps, n_features, latent_depth)
        self.dec = decoder(timesteps, n_features, latent_depth)
        self.disc = discriminator(timesteps, n_features)

        # Print Model summary in console
        self.enc.summary()
        self.dec.summary()
        self.disc.summary()

        # Define Metrics to be printed
        self.metrics_names = [
            "gan_loss",
            "vae_loss",
            "f_dis_loss",
            "r_dis_loss",
            "t_dis_loss",
            "vae_diff_loss",
            "E_loss",
            "D_loss",
            "kl_loss",
            "normal_loss",
            "cl_loss_loc",
            "cl_accuracy_loc",
            "cl_precision_loc",
            "cl_recall_loc",
        ]
        self.metrics = []
        for m in self.metrics_names:
            self.metrics.append(tf.keras.metrics.Mean("m", dtype=tf.float32))

        # Set Training parameter
        lr = 0.0001
        self.enc_optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.dec_optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.disc_optimizer = keras.optimizers.Adam(learning_rate=lr)

        # This is need for random synth data creation
        self.test_lattent_r = tf.random.normal((self.batch_size, self.latent_depth))

        # Load pretrained classificator Model
        self.cl_model_dir = os.path.join(get_model_dir(sd=season_decomp), "model")
        self.cl_model = self.load_eval_classifier()

        # Load a pretrained VAE-GAN Model if weights is given
        if weights:
            self.load_weights(weights_dir, weights)

    @tf.function
    def train_step_vaegan(self, x):
        """
        Vorwärts und Rückwärtspropagation eines Batches
        :param x: Batchdata
        :return:
        """
        lattent_r = tf.random.normal((self.batch_size, self.latent_depth))
        inner_loss_coef = 1
        normal_coef = 0.1
        kl_coef = 0.01
        with tf.GradientTape(persistent=True) as tape:
            lattent, kl_loss = self.enc(x)
            fake = self.dec(lattent)
            dis_fake, dis_inner_fake = self.disc(fake)
            dis_fake_r, _ = self.disc(self.dec(lattent_r))
            dis_true, dis_inner_true = self.disc(x)

            vae_inner = dis_inner_fake - dis_inner_true
            vae_inner = vae_inner * vae_inner

            mean, var = tf.nn.moments(self.enc(x)[0], axes=0)
            var_to_one = var - 1

            normal_loss = tf.reduce_mean(mean * mean) + tf.reduce_mean(
                var_to_one * var_to_one
            )

            kl_loss = tf.reduce_mean(kl_loss)
            vae_diff_loss = tf.reduce_mean(vae_inner)
            f_dis_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    tf.zeros_like(dis_fake), dis_fake
                )
            )
            r_dis_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    tf.zeros_like(dis_fake_r), dis_fake_r
                )
            )
            t_dis_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    tf.ones_like(dis_true), dis_true
                )
            )
            gan_loss = 0.5 * t_dis_loss + 0.25 * f_dis_loss + 0.25 * r_dis_loss
            vae_loss = tf.reduce_mean(tf.abs(x - fake))
            E_loss = vae_diff_loss + kl_coef * kl_loss + normal_coef * normal_loss
            G_loss = inner_loss_coef * vae_diff_loss - gan_loss
            D_loss = gan_loss

        E_grad = tape.gradient(E_loss, self.enc.trainable_variables)
        G_grad = tape.gradient(G_loss, self.dec.trainable_variables)
        D_grad = tape.gradient(D_loss, self.disc.trainable_variables)
        del tape
        self.enc_optimizer.apply_gradients(zip(E_grad, self.enc.trainable_variables))
        self.dec_optimizer.apply_gradients(zip(G_grad, self.dec.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(D_grad, self.disc.trainable_variables))

        return [
            gan_loss,
            vae_loss,
            f_dis_loss,
            r_dis_loss,
            t_dis_loss,
            vae_diff_loss,
            E_loss,
            D_loss,
            kl_loss,
            normal_loss,
        ]

    def train(
        self,
        dataset: tf.data.Dataset,
        epochs: int = 10,
        batch_size: int = 32,
        sample_interval: int = 50,
        test_dataset: tf.data.Dataset = None,
    ):
        """
        Training mit dem dataset
        :param dataset: trainingsdaten
        :param epochs: Anzahl Trainingsdurchläufe
        :param batch_size: Größe des Batches
        :param sample_interval: Interval indem Plot erstellt wird und eine Klassifikation durchgeführt wird
        :param test_dataset: Erstellt synth. Daten basierend auf dem test_datensatz
        :return:
        """
        df_result = pd.DataFrame()
        self.test_dataset = test_dataset

        # Create batches and shuffle
        batched_dataset = dataset.shuffle(int(2 * batch_size)).batch(
            batch_size, drop_remainder=True
        )
        batched_dataset = batched_dataset.shuffle(
            10, reshuffle_each_iteration=True
        ).repeat()

        # Set save_dir
        weights_dir = os.path.join(self.save_dir, "weights")
        e_weights = os.path.join(weights_dir, "encoder")
        g_weights = os.path.join(weights_dir, "generator")
        d_weights = os.path.join(weights_dir, "discriminator")
        check_make_dir(e_weights)
        check_make_dir(g_weights)
        check_make_dir(d_weights)

        epoch = 0
        for batch in batched_dataset:
            # Make forward-backward propagation with batch
            results = self.train_step_vaegan(batch)
            # Make evaluation if sample_interval
            if epoch % sample_interval == 0:
                if self.season_decomp:
                    self.sample_plot_season_decomp(batch, epoch)
                else:
                    self.sample_plot(batch, epoch)
                # Get classification loss
                eval_loss = self.sample_evaluation()
                # Save weights from current Model
                self.disc.save_weights(
                    os.path.join(d_weights, "discriminator_{}.h5".format(epoch))
                )
                self.dec.save_weights(
                    os.path.join(g_weights, "generator_{}.h5".format(epoch))
                )
                self.enc.save_weights(
                    os.path.join(e_weights, "encoder_{}.h5".format(epoch))
                )
                # Log evaluation results
                df_result = pd.concat(
                    [
                        df_result,
                        pd.DataFrame(
                            [eval_loss],
                            index=[epoch],
                            columns=self.metrics_names[-len(eval_loss) :],
                        ),
                    ]
                )
            results = [*results, *eval_loss]

            for metric, result in zip(self.metrics, results):
                metric(result)
            self.print_metrics(epoch)

            epoch += 1
            if epoch == epochs:
                break

        # Save final weights
        self.disc.save_weights(os.path.join(d_weights, "discriminator_final.h5"))
        self.dec.save_weights(os.path.join(g_weights, "generator_final.h5"))
        self.enc.save_weights(os.path.join(e_weights, "encoder_final.h5"))

        # Save logs
        result_fname = os.path.join(self.save_dir, "results.csv")
        df_result.to_csv(result_fname)

        # Create plot of logging
        fig, axes = plt.subplots(2, 2, figsize=(16, 9))
        for col, ax in zip(df_result.columns, axes.flatten()):
            df_result.reset_index().plot(ax=ax, x="index", y=col)

        fig_fname = os.path.join(self.save_dir, "results.png")
        plt.savefig(fig_fname)

    def print_metrics(self, step: int):
        """
        Print logs to console
        :param step:
        :return:
        """
        s = ""
        for name, metric in zip(self.metrics_names, self.metrics):
            s += " " + name + " " + str(np.around(metric.result().numpy(), 3))
        print(f"\rStep : " + str(step) + " " + s, end="", flush=True)
        with self.train_summary_writer.as_default():
            for name, metric in zip(self.metrics_names, self.metrics):
                tf.summary.scalar(name, metric.result().numpy(), step=step)
        for metric in self.metrics:
            metric.reset_states()

    def load_weights(self, weights_dir: str, weights: str):
        """
        Load weights into model
        :param weights_dir: direction of weights
        :param weights: iteration of training
        :return:
        """
        self.disc.load_weights(
            os.path.join(
                weights_dir, "discriminator", "discriminator_{}.h5".format(weights)
            )
        )
        self.dec.load_weights(
            os.path.join(weights_dir, "generator", "generator_{}.h5".format(weights))
        )
        self.enc.load_weights(
            os.path.join(weights_dir, "encoder", "encoder_{}.h5".format(weights))
        )

    def sample_plot_season_decomp(self, x: tuple, epoch: int):
        """
        Create plots for seasonal decomposed data during training
        :param x: batch to plot
        :param epoch:
        :return:
        """
        # Create synth data
        x_new, x_new_r = self.sample_x(x)
        # Create plots
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(16, 10))
        axes = axes.flatten()
        x_numpy = x.numpy()
        random_samples = random.sample(range(len(x[0])), 1)
        # Plot for reconstructed data
        for i, i_sample in enumerate(random_samples):
            df = pd.DataFrame(
                {
                    "seasonal_24": x_numpy[i_sample, :, 0],
                    "seasonal_168": x_numpy[i_sample, :, 1],
                    "resid": x_numpy[i_sample, :, 2],
                    "trend": x_numpy[i_sample, :, 3],
                    "seasonal_24_synth": x_new[i_sample, :, 0],
                    "seasonal_168_synth": x_new[i_sample, :, 1],
                    "resid_synth": x_new[i_sample, :, 2],
                    "trend_synth": x_new[i_sample, :, 3],
                }
            )

            df.reset_index().plot(
                ax=axes[0], x="index", y=["seasonal_24", "seasonal_24_synth"]
            )
            df.reset_index().plot(
                ax=axes[1], x="index", y=["seasonal_168", "seasonal_168_synth"]
            )
            df.reset_index().plot(ax=axes[2], x="index", y=["resid", "resid_synth"])
            df.reset_index().plot(ax=axes[3], x="index", y=["trend", "trend_synth"])

        filename = os.path.join(
            self.save_dir, "training_plots", "plot_{}.jpg".format(epoch)
        )
        check_make_dir(filename, True)
        plt.savefig(filename)
        plt.close(fig)

        # Plot for random synth data
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(14, 8))
        axes = axes.flatten()
        for i in range(0, 1, 1):
            df = pd.DataFrame(
                {
                    "seasonal_24": x_numpy[i_sample, :, 0],
                    "seasonal_168": x_numpy[i_sample, :, 1],
                    "resid": x_numpy[i_sample, :, 2],
                    "trend": x_numpy[i_sample, :, 3],
                    "seasonal_24_synth": x_new_r[i_sample, :, 0],
                    "seasonal_168_synth": x_new_r[i_sample, :, 1],
                    "resid_synth": x_new_r[i_sample, :, 2],
                    "trend_synth": x_new_r[i_sample, :, 3],
                }
            )

            df.reset_index().plot(
                ax=axes[0], x="index", y=["seasonal_24", "seasonal_24_synth"]
            )
            df.reset_index().plot(
                ax=axes[1], x="index", y=["seasonal_168", "seasonal_168_synth"]
            )
            df.reset_index().plot(ax=axes[2], x="index", y=["resid", "resid_synth"])
            df.reset_index().plot(ax=axes[3], x="index", y=["trend", "trend_synth"])

        # Save plots
        filename = os.path.join(
            self.save_dir, "training_plots_random", "plot_{}.jpg".format(epoch)
        )
        check_make_dir(filename, True)
        plt.savefig(filename)
        plt.close(fig)

    def sample_plot(self, x: tuple, epoch: int):
        """
        Create plots for nomal data
        :param x: Batch for sample
        :param epoch:
        :return:
        """
        # Create synth data
        x_new, x_new_r = self.sample_x(x)
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 8))
        axes = axes.flatten()
        x_numpy = x[0].numpy()
        random_samples = random.sample(range(len(x[0])), 6)
        # Plot for reconstructed synth data
        for i, i_sample in enumerate(random_samples):
            df = pd.DataFrame(
                {"vae_gan": x_new[i_sample, :, 0], "original": x_numpy[i_sample, :, 0]}
            )
            df.reset_index().plot(ax=axes[i], x="index", y=["vae_gan", "original"])

        filename = os.path.join(
            self.save_dir, "training_plots", "plot_{}.jpg".format(epoch)
        )
        check_make_dir(filename, True)
        plt.savefig(filename)
        plt.close(fig)

        # Plot for random synth data
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 8))
        axes = axes.flatten()
        for i in range(0, 6, 1):
            df = pd.DataFrame({"vae_gan_random": x_new_r[i, :, 0]})
            df.plot(ax=axes[i], use_index=True, y="vae_gan_random")

        filename = os.path.join(
            self.save_dir, "training_plots_random", "plot_{}.jpg".format(epoch)
        )
        check_make_dir(filename, True)
        plt.savefig(filename)
        plt.close(fig)

    def sample_x(self, x: np.ndarray):
        """
        Create synth data for given batch
        :param x:
        :return:
        """
        lattent, _ = self.enc(x)
        example_data_reconstructed = self.dec(lattent)
        samples = self.dec(self.test_lattent_r)
        return example_data_reconstructed, samples

    def sample_evaluation(self):
        """
        Make trts evaluation during training.
        - Prediction with pretrained classification model for all locations
        :return:
        """
        for x_batch in self.test_dataset:
            # Reconstruct test dataset
            x_fake, _ = self.sample_x(x_batch)
            # Asign label for location
            y_fake = tf.keras.utils.to_categorical(
                np.full(len(x_fake), classification_labels[self.location]), 4
            )
            dataset = tf.data.Dataset.from_tensor_slices((x_fake, y_fake))
            dataset = dataset.batch(len(dataset))
            # Make predictions
            cl_location_classify = self.cl_model.evaluate(dataset, verbose=0)
        return [*cl_location_classify]

    def load_eval_classifier(self):
        """
        Load classifier for evaluation during training
        :return:
        """
        model = tf.keras.models.load_model(self.cl_model_dir)
        return model
