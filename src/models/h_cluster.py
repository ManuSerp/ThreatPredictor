import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from models.cluster_model import ClusterModel;
from tqdm import tqdm

class HClustering(ClusterModel):
    def __init__(self, n_clusters=3, **kwargs):
        super().__init__(**kwargs)
        self.model = AgglomerativeClustering(n_clusters=n_clusters,linkage="single" ,**kwargs)
        self.n_clusters = n_clusters

    def train(self, X):
        print("==== Train begining ====")
        #self.linkage_matrix = linkage(X, method='single') try to allocate 582gb of ram
        self.train_data = X
        return self.model.fit_predict(X)

    def predict(self, X):
        print("==== Predict begining ====")
        result = []
        for i in tqdm(range(len(X))):
            result.append(self.find_closest_cluster(X[i], self.train_data, self.model.labels_))

        return result
    def find_closest_cluster(self,new_data_point, existing_data, clusters):
   
        min_distance = float('inf')
        closest_cluster = None

        # Iterate through each cluster
        for cluster_label in np.unique(clusters):
            # Extract members of this cluster
            cluster_members = existing_data[clusters == cluster_label]

            # Calculate distances from the new data point to each member of the cluster
            distances = np.linalg.norm(cluster_members - new_data_point, axis=1)

            # Find the minimum distance to this cluster
            cluster_min_distance = np.min(distances)

            # Check if this is the closest cluster so far
            if cluster_min_distance < min_distance:
                min_distance = cluster_min_distance
                closest_cluster = cluster_label

        return closest_cluster
    
    def evaluating_clustering_performance(self, X, predict_labels):
        return super().evaluating_clustering_performance(X,predict_labels)

    def plot_dendrogram(self):
        # Plotting the dendrogram
        pass # try to allocate 582gb of ram
        plt.figure(figsize=(10, 7))
        dendrogram(self.linkage_matrix)
        plt.title("Dendrogram")

    def plot(self):
        plt.show()
