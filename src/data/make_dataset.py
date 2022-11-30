"""
Run preprocessing for all types
 - normalize and group by day
 - normalize, season_decompose and group by day
 - season_decompose can take some time
"""
from src.data.preprocessor import Preprocessor

if __name__ == "__main__":
    scale_params = [True, False]
    season_decomp_params = [True, False]
    usecols = ["GlobInc"]
    preprocessor = Preprocessor()

    for scale in scale_params:
        for season_decomp in season_decomp_params:
            preprocessor.make_dataset(scale, season_decomp, usecols)
