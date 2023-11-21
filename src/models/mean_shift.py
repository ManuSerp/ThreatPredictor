import pickle
from sklearn.cluster import MeanShift;
from models.cluster_model import ClusterModel;

class MeanShiftModel(ClusterModel):
    def __init__(self, n_clusters=3, **kwargs):
        super().__init__(**kwargs)
        self.model = MeanShift(n_jobs=-1, **kwargs)

    def train(self, X):
        print("==== Train begining ====")
        self.model.fit(X)

    def predict(self, X):
        print("==== Predict begining ====")
        return self.model.predict(X)
    
    def evaluating_clustering_performance(self, X, labels_pred):
        return super().evaluating_clustering_performance(X, labels_pred)