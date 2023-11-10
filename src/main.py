"""Main code"""
import numpy as np
from models.spectral_clustering import SpectralClusterModel
from models.kmean import KMeanModel
from parser.parser import Parser

def main():
    # Write data variable
    PATH_SAVE="../data/output/temp/"
    LOG_PATH="../data/output/parsing/"

    # Initialize the Preprocess and KMeansModel classes
    parser = Parser(PATH_SAVE,LOG_PATH)
    cluster_model = KMeanModel(n_clusters=3,random_state=0, n_init="auto")  # Adjust parameters as needed

    # Load and preprocess data
    sparse_array,label = parser.parse_kdd('../data/kddcup.data_10_percent')
    # data = parser.parse_file('path/to/data')
    # data = parser.clean_data(data)
    # data = parser.one_hot_encode(data, columns=[1, 2, 3])  # Specify the correct columns

    # Train the cluster model
    cluster_model.train(sparse_array)

    # Make predictions
    predictions = cluster_model.predict(sparse_array)

    # Output or further process the predictions
    print(predictions)

if __name__ == '__main__':
    main()





