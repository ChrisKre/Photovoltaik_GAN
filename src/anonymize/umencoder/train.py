"""
Training für LSTM-Codierer
Davor src/anonymize/umencoder/hyperparametersuche.py ausgeführt werden.
Die Parameter des besten Modells werden automatisch übernommen
"""

import os

import sys

from src.anonymize.umencoder.hyperparametersuche import buil_model_lstm
from src.data.settings import processed_dir, ano_dir
from src.utils import check_make_dir

root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(os.path.join(root))

import keras_tuner
import numpy as np
from tensorflow import keras

import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    # Set Parameter
    location = "Koethen"
    norm_location = "Le_Havre"
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

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    model.summary()

    # # Create Dataset for real training

    # # Create save_dirs
    model_dir = os.path.join(location, "model")
    csv_result_dir = os.path.join(location, "training.csv")
    tensboard_result_dir = os.path.join(location, "tb")
    #
    callbacks = [
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        keras.callbacks.CSVLogger(csv_result_dir, append=True),
        keras.callbacks.TensorBoard(tensboard_result_dir),
    ]

    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=50,
        batch_size=128,
        validation_split=validation_split,
        callbacks=callbacks,
        shuffle=True,
    )
    # Save Model
    model.save(os.path.join(location, "model"))

    # Create Trainingplots
    loss_plot_filename = os.path.join(location, "loss.png")
    plt.figure(figsize=(8, 6))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper right")
    plt.savefig(loss_plot_filename)
    plt.close()

    # Create Synth datasets
    x_test_ano = model.predict(x_test)

    df_ano = pd.DataFrame(np.squeeze(x_test_ano), index=df_orig_location.iloc[-366:].index)

    # Save synth dataset
    fname_ano = os.path.join(ano_dir, "umencoder", f"{location}.csv")
    check_make_dir(fname_ano, True)
    df_ano.to_csv(fname_ano)
