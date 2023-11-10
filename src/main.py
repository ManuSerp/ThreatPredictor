"""Main code"""
import numpy as np
import pickle

from models.spectral_clustering import SpectralClusterModel
from models.kmean import KMeanModel
from models.dbscan import DBSCANModel
from parser.parser import Parser

def main():
    # # Write data variable
    PATH_SAVE="../data/output/temp/"
    LOG_PATH="../data/output/parsing/"

    # # Initialize the Preprocess and KMeansModel classes
    # parser = Parser(PATH_SAVE,LOG_PATH)
    # cluster_model = SpectralClusterModel(n_clusters=2, assign_labels='discretize', random_state=0)
    # cluster_model = KMeanModel(n_clusters=10, random_state=0, n_init="auto")
    cluster_model = DBSCANModel(eps=0.5, min_samples=5)
    # # Load and preprocess data
    with open(LOG_PATH+"sparse.pkl", 'rb') as file:
                sparse_array,label = pickle.load(file)
    
    # data = parser.parse_file('path/to/data')
    # data = parser.clean_data(data)
    # data = parser.one_hot_encode(data, columns=[1, 2, 3])  # Specify the correct columns

    # Train the cluster model
    cluster_model.train(sparse_array)
    predict_labels = cluster_model.predict(sparse_array)

    cluster_model.evaluating_clustering_performance(sparse_array,predict_labels)




if __name__ == '__main__':
    main()





