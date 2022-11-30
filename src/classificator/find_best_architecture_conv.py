"""
Hyperparametersuche für Faltungs-Klassifikator
"""

import os
import sys

root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(os.path.join(root))

import keras_tuner
import numpy as np
from tensorflow import keras

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.data.dataloader import DataLoaderCl
from src.utils import classificator_dir


def buil_model_conv(hp):
    inputs = keras.Input(shape=(24, 1))
    x = inputs
    for i in range(hp.Int("cnn_layers", 1, 5)):
        x = keras.layers.Conv1D(
            filters=hp.Int(f"filters_{i}", min_value=8, max_value=24),
            kernel_size=1,
            activation="relu",
        )(x)
        if hp.Choice("pooling_" + str(i), ["avg", "max"]) == "max":
            x = keras.layers.MaxPool1D(1)(x)
        else:
            x = keras.layers.AvgPool1D(1)(x)

        if hp.Boolean(f"batch_norm_{i}"):
            x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Flatten()(x)

    # A hyperparamter for whether to use dropout layer.
    if hp.Boolean("dropout"):
        x = keras.layers.Dropout(
            hp.Float("dropout_rate", min_value=0.1, max_value=0.5)
        )(x)

    outputs = keras.layers.Dense(4, activation="softmax", name="output")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        metrics=["accuracy"],
    )
    return model


def evaluate_hypermodel(hypermodel, test_dataset, target_names, best_model_result_dir):
    # Evaluation
    y_true = np.array(list(map(lambda y: y[1], test_dataset)))
    x_test = np.array(list(map(lambda y: y[0], test_dataset)))
    y_pred = hypermodel(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    df_cl = pd.DataFrame(
        classification_report(
            np.argmax(y_true, axis=1),
            y_pred,
            target_names=target_names,
            output_dict=True,
        )
    )
    df_cl.to_csv(os.path.join(best_model_result_dir, "classification_report.csv"))
    cm = confusion_matrix(np.argmax(y_true, axis=1), y_pred)
    cm = pd.DataFrame(cm, range(4), range(4))
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        cm,
        annot=True,
        annot_kws={"size": 12},
        fmt="g",
        xticklabels=target_names,
        yticklabels=target_names,
    )  # font size
    plt.savefig(os.path.join(best_model_result_dir, "confusionmatrix.png"))
    df_model_params = pd.DataFrame([best_hps.values])
    df_model_params.to_csv(os.path.join(best_model_result_dir, "df_model_params.csv"))


if __name__ == "__main__":
    x = ["Madrid", "Koethen", "München", "Le_Havre"]
    file_dir = "best_architecure"
    scaled = True
    season_decomp = False
    test_year = 2016
    dl = DataLoaderCl(x, scaled, season_decomp, test_year)
    training_dataset, test_dataset = dl.get_train_test_data()

    validation_split = 0.1

    dataset = training_dataset.shuffle(len(training_dataset))
    # Split into train and val dataset

    x_train = np.array(list(map(lambda x: x[0], dataset)))
    y_train = np.array(list(map(lambda y: y[1], dataset)))

    dir = os.path.join(classificator_dir, "architecture_search_conv")
    project_name = "conv"
    results_dir = os.path.join(dir, "tb")
    tensorboard_dir = os.path.join(dir, "tb_log")
    tuner = keras_tuner.Hyperband(
        buil_model_conv,
        max_epochs=200,
        factor=3,
        objective="val_accuracy",
        directory=results_dir,
        project_name=project_name,
        seed=13,
        hyperband_iterations=3,
    )
    tuner.search(
        x_train,
        y_train,
        validation_split=validation_split,
        epochs=50,
        callbacks=[
            keras.callbacks.TensorBoard(tensorboard_dir),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
        ],
    )

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    model.summary()

    # Create Dataset for real training
    batch_size = 32
    steps_per_epoch = (len(training_dataset) * (1 - validation_split)) / batch_size
    validation_steps = (len(training_dataset) * (validation_split)) / batch_size
    training_dataset = training_dataset.shuffle(len(training_dataset))
    train_size = int((1 - validation_split) * len(training_dataset))
    val_size = int(validation_split * len(training_dataset))
    train_ds = training_dataset.take(train_size)
    val_ds = training_dataset.skip(train_size).take(val_size)
    train_ds = train_ds.batch(batch_size).repeat()
    val_ds = val_ds.batch(batch_size).repeat()

    # Create save_dirs
    best_model_result_dir = os.path.join(dir, project_name, "best")
    csv_result_dir = os.path.join(best_model_result_dir, "training.csv")
    tensboard_result_dir = os.path.join(best_model_result_dir, "tb")

    callbacks = [
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        keras.callbacks.CSVLogger(csv_result_dir, append=True),
        keras.callbacks.TensorBoard(tensboard_result_dir),
    ]

    history = model.fit(
        train_ds,
        epochs=50,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        shuffle=True,
        validation_steps=validation_steps,
    )

    val_acc_per_epoch = history.history["val_accuracy"]
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1

    hypermodel = tuner.hypermodel.build(best_hps)
    # Retrain the model
    history = hypermodel.fit(
        train_ds,
        epochs=best_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        callbacks=callbacks,
        shuffle=True,
        validation_steps=validation_steps,
    )

    hypermodel.save(os.path.join(best_model_result_dir, "model"))

    evaluate_hypermodel(hypermodel, test_dataset, x, best_model_result_dir)
