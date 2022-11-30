"""
Trainiere KLassifikator für paarweise Standorte
Klassifikatoren werden gespeichert unter models/classificator/{locationA_locationB}
"""
import argparse

from src.classificator.classificator import Classificator
from src.data.dataloader import DataLoaderCl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-sd", "--season_decomposed", help="Season_decomposed", type=bool, default=True
    )
    args = parser.parse_args()

    x = ["Madrid", "Koethen", "München", "Le_Havre"]
    scaled = True
    test_year = 2016
    season_decomp = args.season_decomposed

    # Iterate over all locations
    for x_i in x:
        # Iterate over all locations backwards
        for x_j in reversed(x):
            if x_i is x_j:
                break
            # Set parameter
            file_dir = f"{x_i}_{x_j}"

            # Load data for the two locations
            dl = DataLoaderCl([x_i, x_j], scaled, season_decomp, test_year, True)
            training_dataset, test_dataset = dl.get_train_test_data()
            input_shape, num_classes = dl.get_dataset_shapes()
            # Instanziate Classifcator
            cl = Classificator(
                input_dim=input_shape[0],
                num_feat=input_shape[1],
                num_classes=num_classes,
                file_dir=file_dir,
                season_decomp=season_decomp,
            )
            # Train and evaluate Classificator
            cl.train(training_dataset, epochs=50)
            cl.eval(test_dataset, [x_i, x_j])
