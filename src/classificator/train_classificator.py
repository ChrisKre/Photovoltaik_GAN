"""
Trainiere KLassifikator für alle Standorte
"""

import argparse
import os
import sys

root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(os.path.join(root))
from src.data.dataloader import DataLoaderCl
from src.classificator.classificator import Classificator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-sd", "--season_decomposed", help="Season_decomposed", type=bool, default=False
    )
    args = parser.parse_args()

    # Set paramter
    x = ["Madrid", "Koethen", "München", "Le_Havre"]
    file_dir = "all_locations"
    scaled = True
    test_year = 2016
    season_decomp = args.season_decomposed

    # Load data
    dl = DataLoaderCl(x, scaled, season_decomp, test_year)
    training_dataset, test_dataset = dl.get_train_test_data()
    input_shape, num_classes = dl.get_dataset_shapes()

    # Instanziate Classificator
    cl = Classificator(
        input_dim=input_shape[0],
        num_feat=input_shape[1],
        num_classes=num_classes,
        file_dir=file_dir,
        season_decomp=season_decomp,
    )
    # Train and evaluate Classificator
    cl.train(training_dataset, epochs=100)
    cl.eval(test_dataset, x)
