import numpy as np


from tqdm import tqdm





class ClusterLabel:
    def __init__(self,cluster_affect,label,n_clusters):
        self.cluster_affect = cluster_affect
        self.label = label
        self.cluster_distribution = None
        self.cluster_label = None

        # unique labels:
        self.unique_label = np.unique(label)

        self.cluster_distribution = [{}]*self.n_clusters
        


    def cluster_stat(self):
        for i in tqdm(range(self.n_clusters)):
            self.cluster_distribution[i] = {label:0 for label in self.unique_label}

        for i in tqdm(range(len(self.label))):
            self.cluster_distribution[self.cluster_affect[i]][self.label[i]] += 1

        self.cluster_label = [max(self.cluster_distribution[i], key=self.cluster_distribution[i].get) for i in range(self.n_clusters)]

        

        

        


    