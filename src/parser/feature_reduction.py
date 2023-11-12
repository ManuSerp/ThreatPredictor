from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


class FeatureReduction:
    def __init__(self, method='TruncatedSVD'):
        self.method = method
        self.model = None
        self.n_components = None
    
    def fit_transform(self, X):
        if self.model is None:
            raise Exception("Model not fitted. Call create_model first.")
        else:
            if (self.method == 'PCA'):
                X = StandardScaler().fit_transform(X)
                model = self.model
            elif (self.method == 'TruncatedSVD'):
                model = self.model
            X_reduced = model.fit_transform(X)
            return X_reduced

    def create_model(self,n_components):
        if self.method == 'PCA':
            self.model = PCA(n_components=self.n_components, svd_solver='full')
        elif self.method == 'TruncatedSVD':
            self.n_components = n_components
            self.model = TruncatedSVD(n_components=self.n_components)
        else:
            raise ValueError(f"Unknow reduction method: {self.method}")

    
    def plot_variance(self, X, n_components,ratio=0.95):
        """
        Plot the explained variance by each component and the cumulative variance.
        :param X: numpy array of shape (n_samples, n_features)
        :param n_components: number of components to consider for PCA
        """

        if self.method == 'PCA':
            X = StandardScaler().fit_transform(X)
            model = PCA(n_components=n_components)
        elif self.method == 'TruncatedSVD':
            model = TruncatedSVD(n_components=n_components)
        
        model.fit(X)

        n_components_ratio = (model.explained_variance_ratio_.cumsum() >= ratio).argmax() + 1
        print(f"Number of components to expalin {ratio*100}% variance : {n_components_ratio}")

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(model.explained_variance_ratio_) + 1), model.explained_variance_ratio_)
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Component')
        plt.title('Variance Explained by Each Component')

        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(model.explained_variance_ratio_) + 1), np.cumsum(model.explained_variance_ratio_), marker='o')
        plt.ylabel('Cumulative explained variance')
        plt.xlabel('Number of components')
        plt.title('Cumulative Variance Explained')

        plt.tight_layout()
        plt.show()

        return n_components_ratio