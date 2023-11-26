from sklearn.cluster import DBSCAN;
from models.cluster_model import ClusterModel;

class DBSCANModel(ClusterModel):
    def __init__(self, eps=3, n_jobs=-1, **kwargs):
        super().__init__(**kwargs)
        self.model = DBSCAN(eps=eps, **kwargs)

    def train(self, X):
        print("==== Train begining ====")
        self.model.fit(X)

    def predict(self, X):
        print("==== Predict begining ====")
        return self.model.fit_predict(X)
    
    def evaluating_clustering_performance(self, X, predict_labels):
        return super().evaluating_clustering_performance(X,predict_labels)
    