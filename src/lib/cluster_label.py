import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm





class ClusterLabel:
    def __init__(self,cluster_affect,label,n_clusters):
        self.cluster_affect = cluster_affect
        self.label = label
        self.cluster_distribution = None
        self.cluster_label = None
        self.n_clusters = n_clusters

        # unique labels:
        self.unique_label = np.unique(label)

        self.cluster_distribution = [{}]*self.n_clusters
        


    def cluster_stat(self):
        for i in tqdm(range(self.n_clusters)):
            self.cluster_distribution[i] = {label:0 for label in self.unique_label}

        for i in tqdm(range(len(self.label))):
            self.cluster_distribution[self.cluster_affect[i]][self.label[i]] += 1

        self.cluster_label = [max(self.cluster_distribution[i], key=self.cluster_distribution[i].get) for i in range(self.n_clusters)]

        
    def get_predicted_label(self,predict_index):
        res=[]
        for i in predict_index:
            res.append(self.cluster_label[i])
        return res
    
    def calc_stat(self,label,predict_label):
        res = {}
        for i in range(len(label)):
            if label[i] not in res:
                res[label[i]] = {}
                res[label[i]]['total'] = 0
                res[label[i]]['correct'] = 0
            res[label[i]]['total'] += 1
            if label[i] == predict_label[i]:
                res[label[i]]['correct'] += 1
        return res
        
    def plot(self,dict):
        categories = []
        accuracies = []
        for category, values in dict.items():
            total = values['total']
            correct = values['correct']
            accuracy = (correct / total) * 100 if total > 0 else 0
            categories.append(category)
            accuracies.append(accuracy)

        # Creating the bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(categories, accuracies, color='skyblue')
        plt.xlabel('Category')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy of Each Category')
        plt.xticks(rotation=45)
        plt.ylim(0, 110)
        plt.grid(axis='y')

        # Display the plot
        plt.show()
        
        

        


    