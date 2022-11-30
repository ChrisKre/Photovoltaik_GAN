"""
Hyperparametersuche für LSTM-Umkodierer
"""

import os
import sys

from sklearn.model_selection import train_test_split

from src.data.settings import processed_dir

root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(os.path.join(root))

import keras_tuner
import numpy as np
from tensorflow import keras

import pandas as pd


def buil_model_lstm(hp):
    inputs = keras.Input(shape=(24, 1))
    x = inputs
    for i in range(hp.Int("lstm_layers", 1, 5)):
        x = keras.layers.LSTM(
            units=hp.Int(f"units_{i}", min_value=6, max_value=64), return_sequences=True
        )(x)

    outputs = keras.layers.TimeDistributed(
        keras.layers.Dense(1, activation="sigmoid", name="output")
    )(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    hp_learning_rate = hp.Float("learning_rate", min_value=1e-5, max_value=1e-2)
    hp_loss = hp.Choice("loss", values=["mse", "mae"])
    model.compile(
        loss=hp_loss, optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate)
    )
    return model


if __name__ == "__main__":
    # Set Parameter
    location = "Koethen"
    norm_location = "Madrid"
    test_year = 2016

    # Load data to be normed
    fname_location = os.path.join(processed_dir, "scaled", f"{norm_location}.csv")
    df_norm_location = pd.read_csv(fname_location, index_col=0)
    df_norm_location.index = pd.to_datetime(df_norm_location.index)
    # Split into train test
    x_train = df_norm_location[df_norm_location.index.year != test_year]
    x_test = df_norm_location[df_norm_location.index.year == test_year]

    # Load norm location data
    fname_location = os.path.join(processed_dir, "scaled", f"{location}.csv")
    df_orig_location = pd.read_csv(fname_location, index_col=0)
    df_orig_location.index = pd.to_datetime(df_orig_location.index)
    # Split into train test
    y_train = df_orig_location[df_orig_location.index.year != test_year]
    y_test = df_orig_location[df_orig_location.index.year == test_year]

    # Reshape into 3d for lstm-network to work
    x_train = np.expand_dims(x_train.values, 2)
    x_test = np.expand_dims(x_test.values, 2)
    y_train = np.expand_dims(y_train.values, 2)
    y_test = np.expand_dims(y_test.values, 2)

    validation_split = 0.1
    # Set logging parameter
    dir = os.path.join("architecture_search_lstm")
    project_name = "lstm"
    results_dir = os.path.join(dir, "tb")
    tensorboard_dir = os.path.join(dir, "tb_log")

    # Do hyperband search
    tuner = keras_tuner.Hyperband(
        buil_model_lstm,
        max_epochs=200,
        factor=3,
        objective="val_loss",
        directory=results_dir,
        project_name=project_name,
        seed=13,
        hyperband_iterations=3,
    )

    tuner.search(
        x_train,
        y_train,
        validation_split=validation_split,
        epochs=5,
        callbacks=[
            keras.callbacks.TensorBoard(tensorboard_dir),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
        ],
    )

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    model.summary()

    # # Create Dataset for real training

    # # Create save_dirs
    best_model_result_dir = os.path.join(dir, project_name, "best")
    csv_result_dir = os.path.join(best_model_result_dir, "training.csv")
    tensboard_result_dir = os.path.join(best_model_result_dir, "tb")
    #
    callbacks = [
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        keras.callbacks.CSVLogger(csv_result_dir, append=True),
        keras.callbacks.TensorBoard(tensboard_result_dir),
    ]

    x_train, y_train, x_val, y_val = train_test_split(
        x_train, y_train, test_size=0.10, random_state=42
    )
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=50,
        batch_size=128,
        validation_data=(x_val, y_val),
        shuffle=True,
    )
    #
    val_acc_per_epoch = history.history["val_accuracy"]
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    #
    hypermodel = tuner.hypermodel.build(best_hps)
    # # Retrain the model
    history = hypermodel.fit(
        x=x_train,
        y=y_train,
        epochs=50,
        batch_size=128,
        validation_data=(x_val, y_val),
        shuffle=True,
    )
    #
    hypermodel.save(os.path.join(best_model_result_dir, "model"))
    #
