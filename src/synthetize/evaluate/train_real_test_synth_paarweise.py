"""
Durchführung einer binären Klassifikation von synthetischen Daten mit bereits trainierten paarweisen Klassifikatoren.
Dazu muss zunächst 'src/classificator/train_paarweise.py' ausgeführt werden, so dass alle trainierte Klassifikatoren
bereitstehen. Ergebnisse werden defaultmäßig im selben ordner unter 'trts_paarweise/{modelname}' gespeichert
"""

import argparse
import os.path

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from src.data.dataloader import DataLoader
from src.utils import check_make_dir, get_model_dir, get_synth_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-x", "--location", help="Location", type=str, default="Koethen"
    )
    parser.add_argument("-s", "--scaled", help="Scaled", type=bool, default=True)
    parser.add_argument(
        "-sd",
        "--season_decomp",
        help="Make seasonal decomposition",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "-sf",
        "--save_file_dir",
        help="Save Filedir",
        type=str,
        default="trts_paarweise",
    )
    parser.add_argument(
        "-f", "--file_name", help="Filename of fake data", type=str, default=None
    )
    args = parser.parse_args()

    # Set Parameter
    location = args.location
    season_decomp = args.season_decomp
    scaled = args.scaled
    save_file_dir = args.save_file_dir
    fname_fake = args.file_name
    test_year = 2016
    x = ["Madrid", "Koethen", "München", "Le_Havre"]
    x.remove(location)

    # Load fake data, if no file_name given take file from model folder
    dl = DataLoader()
    if fname_fake is None:
        data_dir_fake = get_synth_dir("time_gan", season_decomp)
        data_dir_fake = os.path.join(data_dir_fake, location)
        fname_fake = os.path.join(data_dir_fake, os.listdir(data_dir_fake)[0])
    synth_data = dl.load_to_numpy(fname_fake, season_decomp=season_decomp)

    # Iterate over the necessary Models and make prediction
    for x_i in x:
        # Order of location determines the label, e.g. koethen_madrid = koethen:0, madrid:1
        # Set label according to that schema
        try:
            file_dir = f"{x_i}_{location}"
            models_dir = get_model_dir("classificator", season_decomp, file_dir)
            cl_model = tf.keras.models.load_model(os.path.join(models_dir, "model"))
            y_true = 1
            labels = [x_i, location]
        except:
            file_dir = f"{location}_{x_i}"
            models_dir = get_model_dir("classificator", season_decomp, file_dir)
            cl_model = tf.keras.models.load_model(os.path.join(models_dir, "model"))
            y_true = 0
            labels = [location, x_i]

        # Create dataset for prediction
        y_true = np.full(fill_value=y_true, shape=synth_data.shape[0])
        train_target = tf.keras.utils.to_categorical(y_true, num_classes=2)
        data_predict = tf.data.Dataset.from_tensor_slices(
            (tf.cast(synth_data, tf.float32), train_target)
        )
        data_predict = data_predict.batch(len(data_predict))
        save_dir = os.path.join(save_file_dir, f"{location}{x_i}")

        # Make prediction
        y_pred = cl_model.predict(data_predict)

        # Evaluate prediction
        y_pred = tf.argmax(y_pred, axis=1)
        df_cl = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True))

        cm = confusion_matrix(y_true, y_pred)
        f = sns.heatmap(cm, annot=True, fmt="g", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Vorhergesagte Klasse", fontsize=14)
        plt.ylabel("Tatsächliche Klasse", fontsize=14)
        evaluation_results_fname = os.path.join(save_dir, "results.csv")
        confusion_matrix_fname = os.path.join(save_dir, "confusion_matrix.png")
        check_make_dir(save_dir)
        df_cl.to_csv(evaluation_results_fname)
        plt.savefig(confusion_matrix_fname)
        plt.close()
        print("{} wurde erstellt".format(evaluation_results_fname))
        print("{} wurde erstellt".format(confusion_matrix_fname))
