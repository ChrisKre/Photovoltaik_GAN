"""
Modell für Klassifikator
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras import backend as K
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix, classification_report

from src.utils import check_make_dir, get_model_dir

tf.random.set_seed(1234)


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class Classificator:
    def __init__(
        self,
        input_dim: int = 24,
        num_feat=1,
        num_classes: int = 2,
        file_dir: str = "koethen_madrid",
        season_decomp: bool = True,
        load_dir: str = None,
        eval_dir: str = None,
    ):
        # Set parameter
        self.input_dim = input_dim
        self.num_feat = num_feat
        self.num_classes = num_classes

        # Use pretrained Model saved at load_dir
        if load_dir:
            self.save_dir = load_dir
            self.model = tf.keras.models.load_model(os.path.join(load_dir, "model"))
        else:
            if eval_dir:
                self.save_dir = eval_dir
            else:
                self.save_dir = get_model_dir("classificator", season_decomp, file_dir)
            self.model = self.build_model_lstm(input_dim, num_feat, num_classes)

    def build_model_conv(self, input_dim: int = 24, num_feat=1, num_classes: int = 2):
        """
        Build Discriminative Model from SenseGen Paper
        :return:
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input((input_dim, num_feat), name="input"))
        model.add(tf.keras.layers.Conv1D(filters=15, kernel_size=1, activation="relu"))
        model.add(tf.keras.layers.AvgPool1D(1))
        model.add(tf.keras.layers.Conv1D(filters=21, kernel_size=1, activation="relu"))
        model.add(tf.keras.layers.AvgPool1D(1))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv1D(filters=12, kernel_size=1, activation="relu"))
        model.add(tf.keras.layers.AvgPool1D(1))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv1D(filters=13, kernel_size=1, activation="relu"))
        model.add(tf.keras.layers.AvgPool1D(1))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv1D(filters=1, kernel_size=1, activation="relu"))
        model.add(tf.keras.layers.AvgPool1D(1))
        model.add(tf.keras.layers.Flatten())

        if self.num_classes == 2:
            self.metric = "binary_accuracy"
            self.loss = "categorical_crossentropy"
            model.add(tf.keras.layers.Dense(1, activation="sigmoid", name="output"))
            model.compile(
                loss="binary_crossentropy",
                optimizer=optimizer,
                metrics=[
                    self.metric,
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                ],
            )
        else:
            self.metric = "accuracy"
            self.loss = "categorical_crossentropy"
            model.add(
                tf.keras.layers.Dense(num_classes, activation="softmax", name="output")
            )
            model.compile(
                loss="categorical_crossentropy",
                optimizer=optimizer,
                metrics=[
                    self.metric,
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                ],
            )
        # model.summary()
        return model

    def build_model_lstm(self, input_dim: int = 24, num_feat=1, num_classes: int = 2):
        """
        Build Discriminative Model from SenseGen Paper
        :return:
        """
        optimizer = tf.keras.optimizers.Adam()
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input((input_dim, num_feat), name="input"))
        model.add(tf.keras.layers.LSTM(input_dim, return_sequences=True, name="lstm_1"))
        model.add(
            tf.keras.layers.LSTM(input_dim, return_sequences=False, name="lstm_3")
        )
        model.add(tf.keras.layers.Dropout(0.3))

        if self.num_classes == 2:
            self.metric = "binary_accuracy"
            self.loss = "binary_crossentropy"
            model.add(tf.keras.layers.Dense(1, activation="sigmoid", name="output"))
            model.compile(
                loss=self.loss,
                optimizer=optimizer,
                metrics=[
                    "binary_accuracy",
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                ],
            )
        else:
            self.metric = "accuracy"
            self.loss = "categorical_crossentropy"
            model.add(
                tf.keras.layers.Dense(num_classes, activation="softmax", name="output")
            )
            model.compile(
                loss=self.loss,
                optimizer=optimizer,
                metrics=[
                    "accuracy",
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                ],
            )
        model.summary()
        return model

    def train(
        self,
        dataset: tf.data.Dataset,
        batch_size: int = 32,
        epochs: int = 100,
        validation_split: int = 0.1,
    ):
        steps_per_epoch = (len(dataset) * (1 - validation_split)) / batch_size
        validation_steps = (len(dataset) * (validation_split)) / batch_size

        # Process dataset for training
        dataset = dataset.shuffle(len(dataset))
        # Split into train and val dataset
        train_size = int((1 - validation_split) * len(dataset))
        val_size = int(validation_split * len(dataset))
        train_ds = dataset.take(train_size)
        val_ds = dataset.skip(train_size).take(val_size)
        train_ds = train_ds.batch(batch_size).repeat()
        val_ds = val_ds.batch(batch_size).repeat()
        # Set up callbacks
        csv_result_dir = os.path.join(self.save_dir, "training.log")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
            tf.keras.callbacks.CSVLogger(csv_result_dir, append=True),
        ]
        # Create the directory for log files if not exist already
        check_make_dir(csv_result_dir, True)
        # Start training
        history = self.model.fit(
            train_ds,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            callbacks=callbacks,
            shuffle=True,
            validation_steps=validation_steps,
        )

        # Save the best model from training
        model_filename = os.path.join(self.save_dir, "model")
        self.model.save(model_filename)

        # Create plots from training
        acc_plot_filename = os.path.join(self.save_dir, "acc.png")

        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.plot(history.history[self.metric])
        plt.plot(history.history["val_{}".format(self.metric)])
        plt.ylabel("Accuracy", fontsize=14)
        plt.yticks(fontsize=12)
        plt.xlabel("Epoche", fontsize=14)
        plt.xticks(fontsize=12)
        plt.legend(["Training", "Validierung"], loc="lower right")
        plt.savefig(acc_plot_filename)
        plt.close()

        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        loss_plot_filename = os.path.join(self.save_dir, "loss.png")
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.ylabel(self.loss.replace("_", " ").title(), fontsize=14)
        plt.yticks(fontsize=12)
        plt.xlabel("Epoche", fontsize=14)
        plt.xticks(fontsize=12)
        plt.legend(["Training", "Validierung"], loc="upper right")
        plt.savefig(loss_plot_filename)
        plt.close()

    def eval(self, x: tf.data.Dataset, labels: [str]):
        """
        Evaluate model on test dataset,
        Create confusion matrix
        :param x: test dataset
        :param labels: labels for confusion matrix
        :return:
        """
        labels_optm = []
        for label in labels:
            labels_optm.append(label.replace("_", " ").replace("oe", "ö"))
        # Extract labels from dataset
        y_true = list(map(lambda y: y[1], x))
        # Create one batch for evaluation
        x = x.batch(len(x))

        # Get acutal prediction to calculate recall, ..
        y_pred = self.model.predict(x)
        if self.num_classes != 2:
            y_true = tf.argmax(y_true, axis=1)
            y_pred = tf.argmax(y_pred, axis=1)
        else:
            y_pred = 1 * (y_pred >= 0.5)

        df_cl = pd.DataFrame(
            classification_report(
                y_true, y_pred, target_names=labels_optm, output_dict=True
            )
        )

        cm = confusion_matrix(y_true, y_pred)
        f = sns.heatmap(
            cm, annot=True, fmt="g", xticklabels=labels_optm, yticklabels=labels_optm
        )

        # plt.title('Confusion Matrix')
        plt.xlabel("Vorhergesagte Klasse", fontsize=14)
        plt.ylabel("Tatsächliche Klasse", fontsize=14)

        evaluation_dir = os.path.join(self.save_dir, "evaluation")
        evaluation_results_fname = os.path.join(evaluation_dir, "results.csv")
        confusion_matrix_fname = os.path.join(evaluation_dir, "confusion_matrix.png")
        check_make_dir(evaluation_dir)
        df_cl.to_csv(evaluation_results_fname)
        plt.savefig(confusion_matrix_fname)
        plt.close()
        print("{} wurde erstellt".format(evaluation_results_fname))
        print("{} wurde erstellt".format(confusion_matrix_fname))

    def sample_evaluation_training(
        self,
        dataset: tf.data.Dataset,
        batch_size: int = 32,
        epochs: int = 50,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
    ):
        """
        Used in VAE-GAN for evaluation during training
        Used to learn to distinguish between real-fake data generated from VAE-GAN
        :param dataset:
        :param batch_size:
        :param epochs:
        :param train_split:
        :param val_split:
        :param test_split:
        :return:
        """
        steps_per_epoch = (len(dataset) * (1 - val_split)) / batch_size
        validation_steps = (len(dataset) * (val_split)) / batch_size

        # Process dataset for training
        dataset = dataset.shuffle(len(dataset))

        # Split into train and val dataset
        train_size = int(train_split * len(dataset))
        val_size = int(val_split * len(dataset))
        train_ds = dataset.take(train_size)
        val_ds = dataset.skip(train_size).take(val_size)
        test_ds = dataset.skip(train_size).skip(val_size)
        # Make batches
        train_ds = train_ds.batch(batch_size).repeat()
        val_ds = val_ds.batch(batch_size).repeat()

        self.model.fit(
            train_ds,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            shuffle=True,
            validation_steps=validation_steps,
            verbose=0,
        )

        # y_true = list(map(lambda y: y[1], test_ds))
        test_ds = test_ds.batch(len(test_ds))
        # evaluate the model
        loss, accuracy, precision, recall = self.model.evaluate(test_ds, verbose=0)
        return [loss, accuracy, recall, precision]
