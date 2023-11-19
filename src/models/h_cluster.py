from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from models.cluster_model import ClusterModel;

class AgglomerativeClustering(ClusterModel):
    def __init__(self, n_clusters=3, **kwargs):
        super().__init__(**kwargs)
        self.model = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
        self.n_clusters = n_clusters

    def train(self, X):
        print("==== Train begining ====")
        self.model.fit(X)
        self.linkage_matrix = linkage(X, method='ward')

    def predict(self, X):
        print("==== Predict begining ====")
        return self.model.predict(X)
    
    def evaluating_clustering_performance(self, X, predict_labels):
        return super().evaluating_clustering_performance(X,predict_labels)

    def plot_dendrogram(self):
        # Plotting the dendrogram
        plt.figure(figsize=(10, 7))
        dendrogram(self.linkage_matrix)
        plt.title("Dendrogram")
