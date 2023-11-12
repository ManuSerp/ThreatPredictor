"""Main code"""
import numpy as np
from models.spectral_clustering import SpectralClusterModel
from models.kmean import KMeanModel
from models.dbscan import DBSCANModel
from parser.parser import Parser
from parser.feature_reduction import FeatureReduction


def main():
    # Write data variable
    PATH_SAVE="../data/output/temp/"
    LOG_PATH="../data/output/parsing/"

    # Initialize the Preprocess, FeatureReduction and KMeansModel classes
    parser = Parser(PATH_SAVE,LOG_PATH)
    feature_reduction = FeatureReduction('TruncatedSVD')
    cluster_model = KMeanModel()
    # cluster_model = SpectralClusterModel(n_clusters=2, assign_labels='discretize', random_state=0)
    # cluster_model = DBSCANModel(eps=3, min_samples=2)

    # Load and preprocess data
    sparse_array,label = parser.parse_kdd('../data/kddcup.data_10_percent')
    # data = parser.parse_file('path/to/data')
    # data = parser.clean_data(data)
    # data = parser.one_hot_encode(data, columns=[1, 2, 3])  # Specify the correct columns
    
    # Feature reduction
    n_components_ratio = feature_reduction.plot_variance(sparse_array, sparse_array.shape[1], 0.95)
    feature_reduction.create_model(n_components_ratio)
    data = feature_reduction.fit_transform(sparse_array)

    # Train the cluster model
    data = cluster_model.train(data)
    predict_labels = cluster_model.predict(data)

    cluster_model.evaluating_clustering_performance(data,predict_labels)




if __name__ == '__main__':
    main()





