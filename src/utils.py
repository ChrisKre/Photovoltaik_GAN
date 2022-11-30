"""
Helferfunktionen für Filehandling
"""
import logging
import os
import pathlib

from src.data.settings import synth_dir

project_dir = os.path.dirname(os.path.join(pathlib.Path(__file__).parent.resolve()))
data_dir = os.path.join(project_dir, "data")
plot_dir = os.path.join(project_dir, "plots")


def check_make_dir(path: str, file: bool = False):
    """
    Create directory for given path if not yet exists
    :param path:
    :return:
    """
    if file:
        dir = os.path.dirname(path)
    else:
        dir = path
    if not os.path.exists(dir):
        os.makedirs(dir)
        logging.debug("{} wurde erstellt".format(dir))


def build_dir(model_dir, sd):
    model_dir = (
        os.path.join(model_dir, "season_decomposed")
        if sd
        else os.path.join(model_dir, "normal")
    )
    return model_dir


def get_model_dir(
    type: str = "classificator", sd: bool = True, file_dir=None, ano=False
):
    """
    Returns direction for model
    :param type:
    :param sd:
    :param file_dir:
    :param ano:
    :return:
    """
    if type == "classificator":
        model_dir = classificator_dir
        model_dir = os.path.join(model_dir, "lstm")

    elif type == "time_gan":
        model_dir = time_gan_dir
    else:
        model_dir = vae_gan_dir
        if ano:
            model_dir = ano_dir

    model_dir = build_dir(model_dir, sd)

    if type == "classificator":
        if file_dir:
            model_dir = os.path.join(model_dir, file_dir)
        else:
            model_dir = os.path.join(model_dir, "all_locations")
    return model_dir


def get_synth_dir(type: str = "time_gan", sd: bool = True):
    """
    Returns direction of synth data for model and preprocessing
    :param type:
    :param sd:
    :return:
    """
    if type == "time_gan":
        synth_data_dir = os.path.join(synth_dir, "time_gan")
    else:
        synth_data_dir = os.path.join(synth_dir, "vae_gan")

    synth_data_dir = build_dir(synth_data_dir, sd)
    return synth_data_dir


def get_data_dir(dir_in: str, scaled: bool = True, sd: bool = True):
    """
    Returns direction for preprocessing
    :param dir_in:
    :param scaled:
    :param sd:
    :return:
    """
    if scaled == True:
        dir_in = os.path.join(dir_in, "scaled")
    if sd == True:
        dir_in = os.path.join(dir_in, "season_decomposed")
    return dir_in


models_dir = os.path.join(project_dir, "models")
classificator_dir = os.path.join(models_dir, "classificator")
vae_gan_dir = os.path.join(models_dir, "synthetize", "vae_gan")
ano_dir = os.path.join(models_dir, "ano", "vae_gan")
time_gan_dir = os.path.join(models_dir, "synthetize", "time_gan")
seasonal_decomposed_cols = ["seasonal_24", "seasonal_8760", "resid", "trend"]
