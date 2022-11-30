import os

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from src.data.settings import classification_labels
from src.utils import check_make_dir, seasonal_decomposed_cols


def classifier_predict(
    model: tf.keras.models.Model, x: np.ndarray, location: int, save_dir: str
):
    """
    Predict with given model
    :param model:
    :param x:
    :param location:
    :param save_dir:
    :return:
    """
    a = np.empty(len(x))
    a.fill(classification_labels[location])
    y_true = tf.keras.utils.to_categorical(a, len(classification_labels))
    # Get acutal prediction to calculate recall, ..
    y_pred = model.predict(x)
    y_true = tf.argmax(y_true, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)

    pred_labels = []
    for i in classification_labels.keys():
        if classification_labels[i] in np.unique(y_pred.numpy()):
            pred_labels.append(i)

    df_cl = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            target_names=pred_labels,
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


def synth_to_df(season_decomp: bool, synth_data: np.ndarray):
    """
    Convert data to dataframe
    :param season_decomp:
    :param synth_data:
    :return:
    """
    if season_decomp:
        df = pd.DataFrame(
            {
                seasonal_decomposed_cols[0]: synth_data[:, :, 0].tolist(),
                seasonal_decomposed_cols[1]: synth_data[:, :, 1].tolist(),
                seasonal_decomposed_cols[2]: synth_data[:, :, 2].tolist(),
                seasonal_decomposed_cols[3]: synth_data[:, :, 3].tolist(),
            }
        )
    else:
        df = pd.DataFrame(np.squeeze(synth_data))
    return df


def sample_evaluation(model: tf.keras.models.Model, x_fake: np.ndarray, location: str):
    """
    Make classifier prediction with given model for given synth data
    :param model:
    :param x_fake:
    :param location:
    :return:
    """
    y_fake = tf.keras.utils.to_categorical(
        np.full(len(x_fake), classification_labels[location]), 4
    )
    dataset = tf.data.Dataset.from_tensor_slices((x_fake, y_fake))
    dataset = dataset.batch(len(dataset))
    cl_location_classify = model.evaluate(dataset, verbose=0)

    return [*cl_location_classify]


def auto_correlation(data):
    """
    Calculate Autokorrelation
    :param data:
    :return:
    """
    out = []
    for d in data:
        # Mean
        mean = np.mean(d)

        # Variance
        var = np.var(d)

        # Normalized data
        ndata = d - mean

        acorr = np.correlate(ndata, ndata, "full")[len(ndata) - 1 :]
        acorr = acorr / var / len(ndata)
        out.append(acorr)
    return out


def auto_correlation_normal(data):
    """
    Calculate Autokorrelation
    :param data:
    :return:
    """
    # Mean
    mean = np.mean(data)

    # Variance
    var = np.var(data)

    # Normalized data
    ndata = data - mean

    acorr = np.correlate(ndata, ndata, "full")[len(ndata) - 1 :]
    acorr = acorr / var / len(ndata)
    return acorr
