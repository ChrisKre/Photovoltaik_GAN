import argparse
import os.path

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from src.data.settings import ano_dir, classification_labels
from src.utils import check_make_dir, get_model_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--location", help="Location", type=str, default="Madrid")
    parser.add_argument(
        "-y", "--norm_location", help="Location", type=str, default="Koethen"
    )

    args = parser.parse_args()

    location = args.location
    norm_location = args.norm_location

    fname_location = os.path.join(ano_dir, "umencoder", f"{location}.csv")
    df_location = pd.read_csv(fname_location, index_col=0)
    df_location.index = pd.to_datetime(df_location.index)
    df_location_test = df_location[df_location.index.year == 2016]
    location_test = np.expand_dims(df_location.values, 2)

    save_dir = os.path.join("classfier_prediction", f"{location}_zu_{norm_location}")

    eval_classificator_dir = os.path.join(get_model_dir(sd=False), "model")
    eval_classificator_model = tf.keras.models.load_model(eval_classificator_dir)

    a = np.empty(len(location_test))
    a.fill(classification_labels[norm_location])
    y_true = tf.keras.utils.to_categorical(a, len(classification_labels))
    # Get acutal prediction to calculate recall, ..
    y_pred = eval_classificator_model.predict(location_test)
    y_true = tf.argmax(y_true, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)

    df_cl = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            target_names=classification_labels,
            output_dict=True,
            labels=np.unique(y_pred.numpy()),
        )
    )

    cm = confusion_matrix(y_true, y_pred)
    f = sns.heatmap(
        cm,
        annot=True,
        fmt="g",
        xticklabels=classification_labels,
        yticklabels=classification_labels,
    )

    # plt.title('Confusion Matrix')
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
