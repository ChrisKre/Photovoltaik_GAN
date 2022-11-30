"""
Merge the raw datasets
"""
from src.data.preprocessor import Preprocessor

if __name__ == "__main__":
    preprocessor = Preprocessor()
    preprocessor.merge_raw()
