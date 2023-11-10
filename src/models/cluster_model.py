from sklearn import metrics;

class ClusterModel:
    def __init__(self, **kwargs):
        pass

    def train(self, X):
        raise NotImplementedError("Train method must be implemented")

    def predict(self, X):
        raise NotImplementedError("Predict method must be implemented")
    
    def evaluating_clustering_performance(self,X,predict_labels):
        labels = self.model.labels_
        labels_pred = predict_labels
        print(f"Estimated number of clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")
        print(f"Estimated number of noise points: {list(labels).count(-1)}")
        print(f"Homogeneity: {metrics.homogeneity_score(labels, labels_pred):.3f}")
        print(f"Completeness: {metrics.completeness_score(labels, labels_pred):.3f}")
        print(f"V-measure: {metrics.v_measure_score(labels, labels_pred):.3f}")
        print(f"Adjusted Rand Index: {metrics.adjusted_rand_score(labels, labels_pred):.3f}")
        print(
            "Adjusted Mutual Information:"
            f" {metrics.adjusted_mutual_info_score(labels, labels_pred):.3f}"
        )
        # print(f"Silhouette Coefficient: {metrics.silhouette_score(X, labels):.3f}")
