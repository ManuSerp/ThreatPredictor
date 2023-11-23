import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score




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
        
    """"
    1. Initialization of Cluster Distribution: 
       For each cluster, it initializes a dictionary that tracks the count of each unique label present in the cluster. 
       This is achieved by iterating over the total number of clusters (self.n_clusters) and setting up a dictionary 
       for each cluster with all unique labels initialized to a count of zero.
    2. Counting Labels in Each Cluster: 
       The function then iterates over all the data points (labels) and updates the count of the corresponding label 
       in the appropriate cluster (as indicated by self.cluster_affect, which holds the cluster assignment for each data point).
       This step effectively calculates how many times each label occurs in each cluster.
    
    The resultant self.cluster_distribution provides view of the label composition of each cluster,
    while self.cluster_label gives a quick reference to the most characteristic label of each cluster.
    """

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
    
    """
    Calculate metrics like precisoin, recall and F1-score, 
    """
    def calc_metrics(self, labels, predict_labels):
        res = {}

        for cluster in self.unique_label: 
                res[cluster] = {
                    'precision': precision_score(labels, predict_labels, labels=[cluster], average=None),
                    # 'recall': recall_score(labels, predict_labels, labels=[cluster], average=None),
                    'f1_score': f1_score(labels, predict_labels, labels=[cluster], average=None)
                }
        return res
        
    def plot(self,metrics_dict,dict):
        categories = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []

        for category, values in metrics_dict.items():
            categories.append(category)
            precisions.extend(values['precision'])
            # recalls.extend(values['recall'])
            f1_scores.extend(values['f1_score'])
            total = dict[category]['total']
            correct = dict[category]['correct']
            accuracy = correct/total if total > 0 else 0
            accuracies.append(accuracy)

        # Creating the bar chart
        x = np.arange(len(categories))
        width = 0.2

        print(precisions)

        plt.figure(figsize=(12, 6))
        # plt.bar(x - 1.5*width, precisions, width, label='Precision')
        # plt.bar(x, - 0.5*width, recalls, label='Recall')
        # plt.bar(x + 0.5*width, f1_scores, width, label='F1 Score')
        # plt.bar(x + 1.5*width, accuracies, width, label='Accuracy')
        plt.bar (x - width, precisions, width, label='Precison')
        plt.bar(x, f1_scores, width, label='Precision')
        plt.bar(x + width, accuracies, width, label='Accuracy')

        plt.xlabel('Category')
        plt.ylabel('Scores (%)')
        plt.title('Accuracy, Precision and F1 Score for Each Category')
        plt.xticks(x, categories, rotation=45)
        plt.ylim(0, 1.1)
        plt.grid(axis='y')

        # Display the plot
        plt.show()
        
        

        


    