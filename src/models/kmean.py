from sklearn.cluster import KMeans;
from models.cluster_model import ClusterModel;

class KMeanModel(ClusterModel):
    def __init__(self, n_clusters=3, **kwargs):
        super().__init__(**kwargs)
        self.model = KMeans(n_clusters=n_clusters, **kwargs)

    def train(self, X):
        print("==== Train begining ====")
        self.model.fit(X)

    def predict(self, X):
        print("==== Predict begining ====")
        return self.model.predict(X)
    
    def evaluating_clustering_performance(self, X, predict_labels):
        return super().evaluating_clustering_performance(X,predict_labels)
    