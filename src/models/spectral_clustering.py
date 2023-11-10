import pickle
from sklearn.cluster import SpectralClustering;
from models.cluster_model import ClusterModel;

class SpectralClusterModel(ClusterModel):
    def __init__(self, n_clusters=3, **kwargs):
        super().__init__(**kwargs)
        self.model = SpectralClustering(n_clusters=n_clusters, **kwargs)

    def train(self, X):
        print("==== Train begining ====")
        self.model.fit(X)

    def predict(self, X):
        print("==== Predict begining ====")
        return self.model.predict(X)