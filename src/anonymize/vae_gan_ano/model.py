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
    mean, logsigma = args
    epsilon = keras.backend.random_normal(shape=keras.backend.shape(mean))
    return mean + tf.exp(logsigma / 2) * epsilon


def encoder(timesteps: int = 24, n_features: int = 4, latent_depth: int = 3):
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
        location: str = "Madrid",
        season_decomp: bool = False,
        norm_location: str = "Koethen",
        disc_class: bool = False,
    ):

        self.season_decomp = season_decomp
        self.save_dir = get_model_dir("vae_gan", season_decomp, ano=True)
        self.save_dir = os.path.join(self.save_dir, f"{location}_zu_{norm_location}")
        check_make_dir(self.save_dir)

        self.logging_dir = os.path.join(self.save_dir, "training_log")
        self.train_summary_writer = tf.summary.create_file_writer(self.logging_dir)

        self.timesteps = timesteps
        self.n_features = n_features
        self.latent_depth = latent_depth
        self.batch_size = batch_size
        self.location = location
        self.norm_location = norm_location

        self.enc = encoder(timesteps, n_features, latent_depth)
        self.dec = decoder(timesteps, n_features, latent_depth)
        self.disc = discriminator(timesteps, n_features)

        self.enc.summary()
        self.dec.summary()
        self.disc.summary()

        self.metrics_names = [
            "gan_loss",
            "vae_loss",
            "f_dis_loss",
            "r_dis_loss",
            "t_dis_loss",
            "vae_diff_loss",
            "E_loss",
            "D_loss",
            "G_loss",
            "kl_loss",
            "normal_loss",
            "fake_classifiction_loss",
            "cl_loss_loc",
            "cl_accuracy_loc",
            "cl_precision_loc",
            "cl_recall_loc",
        ]

        if disc_class == True:
            self.metrics_names.remove("fake_classifiction_loss")

        self.metrics = []
        for m in self.metrics_names:
            self.metrics.append(tf.keras.metrics.Mean("m", dtype=tf.float32))

        lr = 0.0001
        self.enc_optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.dec_optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.disc_optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.test_lattent_r = tf.random.normal((self.batch_size, self.latent_depth))

        self.cl_model_dir = os.path.join(get_model_dir(sd=season_decomp), "model")
        self.cl_model = self.load_eval_classifier()
        self.classification_loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False
        )
        self.classifcation_label = classification_labels[norm_location]
        y_true = tf.fill(dims=(batch_size, 1), value=self.classifcation_label)
        self.y_true = tf.keras.utils.to_categorical(y_true, num_classes=4)

        if weights:
            self.load_weights(weights_dir, weights)

    @tf.function
    def train_step_vaegan_classification(self, x):
        lattent_r = tf.random.normal((self.batch_size, self.latent_depth))
        inner_loss_coef = 1
        normal_coef = 0.1
        kl_coef = 0.01

        with tf.GradientTape(persistent=True) as tape:
            lattent, kl_loss = self.enc(x)
            fake = self.dec(lattent)
            dis_fake, dis_inner_fake = self.disc(fake)
            dis_fake_r, _ = self.disc(self.dec(lattent_r))
            # Make transform here
            dis_true, dis_inner_true = self.disc(x)
            fake_classifiction = self.cl_model(fake)

            fake_classifiction_loss = self.classification_loss(
                self.y_true, fake_classifiction
            )

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
            G_loss = (
                inner_loss_coef * vae_diff_loss - gan_loss
            ) + 0.025 * fake_classifiction_loss
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
            G_loss,
            kl_loss,
            normal_loss,
            fake_classifiction_loss,
        ]

    def train_cl(
        self,
        dataset: tf.data.Dataset,
        epochs: int = 10,
        batch_size: int = 32,
        sample_interval: int = 50,
        test_dataset: tf.data.Dataset = None,
    ):
        df_result = pd.DataFrame()
        self.test_dataset = test_dataset
        batched_dataset = dataset.shuffle(int(2 * batch_size)).batch(
            batch_size, drop_remainder=True
        )
        batched_dataset = batched_dataset.shuffle(
            10, reshuffle_each_iteration=True
        ).repeat()
        # todo:
        epoch = 0
        weights_dir = os.path.join(self.save_dir, "weights")
        e_weights = os.path.join(weights_dir, "encoder")
        g_weights = os.path.join(weights_dir, "generator")
        d_weights = os.path.join(weights_dir, "discriminator")
        check_make_dir(e_weights)
        check_make_dir(g_weights)
        check_make_dir(d_weights)

        for batch in batched_dataset:
            results = self.train_step_vaegan_classification(batch)
            if epoch % sample_interval == 0:
                if self.season_decomp:
                    self.sample_plot_season_decomp(batch, epoch)
                else:
                    self.sample_plot(batch, epoch)
                eval_loss = self.sample_evaluation()
                self.disc.save_weights(
                    os.path.join(d_weights, "discriminator_{}.h5".format(epoch))
                )
                self.dec.save_weights(
                    os.path.join(g_weights, "generator_{}.h5".format(epoch))
                )
                self.enc.save_weights(
                    os.path.join(e_weights, "encoder_{}.h5".format(epoch))
                )
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

        self.disc.save_weights(os.path.join(d_weights, "discriminator_final.h5"))
        self.dec.save_weights(os.path.join(g_weights, "generator_final.h5"))
        self.enc.save_weights(os.path.join(e_weights, "encoder_final.h5"))
        result_fname = os.path.join(self.save_dir, "results.csv")
        df_result.to_csv(result_fname)

        fig, axes = plt.subplots(2, 2, figsize=(16, 9))
        for col, ax in zip(df_result.columns, axes.flatten()):
            df_result.reset_index().plot(ax=ax, x="index", y=col)

        fig_fname = os.path.join(self.save_dir, "results.png")
        plt.savefig(fig_fname)

    @tf.function
    def train_step_vaegan_disc(self, x, y):
        lattent_r = tf.random.normal((self.batch_size, self.latent_depth))
        inner_loss_coef = 1
        normal_coef = 0.1
        kl_coef = 0.01

        with tf.GradientTape(persistent=True) as tape:
            lattent, kl_loss = self.enc(x)
            fake = self.dec(lattent)
            dis_fake, dis_inner_fake = self.disc(fake)
            dis_fake_r, _ = self.disc(self.dec(lattent_r))
            # Make transform here
            dis_true, dis_inner_true = self.disc(y)

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
            vae_loss = tf.reduce_mean(tf.abs(y - fake))
            E_loss = vae_diff_loss + kl_coef * kl_loss + normal_coef * normal_loss
            G_loss = (inner_loss_coef * vae_diff_loss - gan_loss) + 0.025 * vae_loss
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
            G_loss,
            kl_loss,
            normal_loss,
        ]

    def train(
        self,
        dataset: tf.data.Dataset,
        norm_dataset: tf.data.Dataset,
        epochs: int = 10,
        batch_size: int = 32,
        sample_interval: int = 50,
        test_dataset: tf.data.Dataset = None,
    ):

        df_result = pd.DataFrame()
        self.test_dataset = test_dataset
        batched_dataset = dataset.shuffle(int(2 * batch_size)).batch(
            batch_size, drop_remainder=True
        )
        batched_dataset = batched_dataset.shuffle(
            10, reshuffle_each_iteration=True
        ).repeat()

        batched_norm_dataset = norm_dataset.shuffle(int(2 * batch_size)).batch(
            batch_size, drop_remainder=True
        )
        batched_norm_dataset = batched_norm_dataset.shuffle(
            10, reshuffle_each_iteration=True
        ).repeat()
        # todo:
        epoch = 0
        weights_dir = os.path.join(self.save_dir, "weights")
        e_weights = os.path.join(weights_dir, "encoder")
        g_weights = os.path.join(weights_dir, "generator")
        d_weights = os.path.join(weights_dir, "discriminator")
        check_make_dir(e_weights)
        check_make_dir(g_weights)
        check_make_dir(d_weights)

        for batch, norm_batch in zip(batched_dataset, batched_norm_dataset):
            results = self.train_step_vaegan_disc(batch, norm_batch)
            if epoch % sample_interval == 0:
                if self.season_decomp:
                    self.sample_plot_season_decomp(batch, epoch)
                else:
                    self.sample_plot(batch, epoch)
                eval_loss = self.sample_evaluation()
                self.disc.save_weights(
                    os.path.join(d_weights, "discriminator_{}.h5".format(epoch))
                )
                self.dec.save_weights(
                    os.path.join(g_weights, "generator_{}.h5".format(epoch))
                )
                self.enc.save_weights(
                    os.path.join(e_weights, "encoder_{}.h5".format(epoch))
                )
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

        self.disc.save_weights(os.path.join(d_weights, "discriminator_final.h5"))
        self.dec.save_weights(os.path.join(g_weights, "generator_final.h5"))
        self.enc.save_weights(os.path.join(e_weights, "encoder_final.h5"))
        result_fname = os.path.join(self.save_dir, "results.csv")
        df_result.to_csv(result_fname)

        fig, axes = plt.subplots(2, 2, figsize=(16, 9))
        for col, ax in zip(df_result.columns, axes.flatten()):
            df_result.reset_index().plot(ax=ax, x="index", y=col)

        fig_fname = os.path.join(self.save_dir, "results.png")
        plt.savefig(fig_fname)

    def print_metrics(self, step):
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
        x_new, x_new_r = self.sample_x(x)
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(16, 10))
        axes = axes.flatten()
        x_numpy = x.numpy()
        random_samples = random.sample(range(len(x[0])), 1)
        # todo: PLOT
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

        filename = os.path.join(
            self.save_dir, "training_plots_random", "plot_{}.jpg".format(epoch)
        )
        check_make_dir(filename, True)
        plt.savefig(filename)
        plt.close(fig)

    def sample_plot(self, x: tuple, epoch: int):
        x_new, x_new_r = self.sample_x(x)
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 8))
        axes = axes.flatten()
        x_numpy = x[0].numpy()
        random_samples = random.sample(range(len(x[0])), 6)
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

    def sample_x(self, x):
        lattent, _ = self.enc(x)
        example_data_reconstructed = self.dec(lattent)
        samples = self.dec(self.test_lattent_r)
        return example_data_reconstructed, samples

    def sample_evaluation(self):
        for x_batch in self.test_dataset:
            x_fake, _ = self.sample_x(x_batch)
            y_fake = tf.keras.utils.to_categorical(
                np.full(len(x_fake), classification_labels[self.norm_location]), 4
            )
            dataset = tf.data.Dataset.from_tensor_slices((x_fake, y_fake))
            dataset = dataset.batch(len(dataset))
            cl_location_classify = self.cl_model.evaluate(dataset, verbose=0)
        return [*cl_location_classify]

    def load_eval_classifier(self):
        model = tf.keras.models.load_model(self.cl_model_dir)
        return model
