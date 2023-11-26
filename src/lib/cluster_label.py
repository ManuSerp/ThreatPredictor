import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
import random
import seaborn as sns

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
                    'recall': recall_score(labels, predict_labels, labels=[cluster], average=None),
                    'f1_score': f1_score(labels, predict_labels, labels=[cluster], average=None)
                }
        return res
        
    def plot_metrics(self,metrics_dict,dict):
        categories = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []

        for category, values in metrics_dict.items():
            categories.append(category)
            precisions.extend(values['precision'])
            recalls.extend(values['recall'])
            f1_scores.extend(values['f1_score'])
            if (category in dict):
                total = dict[category]['total']
                correct = dict[category]['correct']
                accuracy = correct/total if total > 0 else 0
            else:
                accuracy = -0.1
            accuracies.append(accuracy)

        # Creating the bar chart
        x = np.arange(len(categories))
        width = 0.2

        plt.figure(figsize=(12, 6))
        plt.bar(x - 1.5*width, precisions, width, label='Precision')
        plt.bar(x, - 0.5*width, recalls, label='Recall')
        plt.bar(x + 0.5*width, f1_scores, width, label='F1 Score')
        plt.bar(x + 1.5*width, accuracies, width, label='Accuracy')

        plt.xlabel('Category')
        plt.ylabel('Scores (%)')
        plt.title('Accuracy, Recall, Precision and F1 Score for Each Category')
        plt.xticks(x, categories, rotation=45)
        plt.ylim(-0.1, 1.1)
        plt.grid(axis='y')
        plt.legend()

        # Display the plot
        plt.show()


    def calc_anomaly_ratio(self,label,predict_label):
        res = {}
        res["normal."] = {}
        res["normal."]['total'] = 0
        res["normal."]['correct'] = 0
        res["anomaly"] = {}
        res["anomaly"]['total'] = 0
        res["anomaly"]['correct'] = 0

        for i in range(len(label)):
            if label[i] == "normal.":
                res["normal."]['total'] += 1
                if label[i] == predict_label[i]:
                    res["normal."]['correct'] += 1
            else:
                res["anomaly"]['total'] += 1
                if predict_label[i] != "normal.":
                    res["anomaly"]['correct'] += 1
    
        return res
    
    def plot(self, stats_dict):
        categories = []
        accuracies = []
        counts = []
        for category, values in stats_dict.items():
            total = values['total']
            correct = values['correct']
            accuracy = (correct / total) * 100 if total > 0 else 0

            categories.append(category)
            accuracies.append(accuracy)
            counts.append(total)

        # Creating the subplot layout
        fig, axs = plt.subplots(2, 1, figsize=(12, 12))  # Two rows, one column

        # Plotting accuracy
        axs[0].bar(categories, accuracies, color='skyblue')
        axs[0].set_xlabel('Category')
        axs[0].set_ylabel('Accuracy (%)')
        axs[0].set_title('Accuracy of Each Category')
        axs[0].set_xticklabels(categories, rotation=45)
        axs[0].set_ylim(0, 110)
        axs[0].grid(axis='y')

        # Plotting count
        axs[1].bar(categories, counts, color='lightgreen')
        axs[1].set_xlabel('Category')
        axs[1].set_ylabel('Count')
        axs[1].set_title('Count of Each Category')
        axs[1].set_xticklabels(categories, rotation=45)
        axs[1].set_ylim(0, max(counts) + 10)  # Adjust the y-axis limit
        axs[1].grid(axis='y')

        # Display the plot
        plt.tight_layout()
        plt.show()

    def plot_anomaly_ratio(self, anomaly_ratio):
        categories = []
        ratios = []

        for category, dic in anomaly_ratio.items():
            # Calculate the ratio
            ratio = dic['correct'] / dic['total'] if dic['total'] > 0 else 0
            ratio=ratio*100
            print(f"Anomaly detection analysis:\n{category}: {ratio:.2f}%")
            categories.append(category)
            ratios.append(ratio)

        # Create the bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(categories, ratios, color='purple')
        plt.xlabel('Category')
        plt.ylabel('Anomaly Ratio')
        plt.title('Anomaly Ratio by Category')
        plt.xticks(rotation=45)
        plt.ylim(0, 100)  # Ratio is between 0 and 1
        plt.grid(axis='y')

    def plot_combined_pairwise(self, data_out, labels, features, sample_size_by_cat, save_path):
        # Check if the sample size is larger than the dataset
        indices = []
        if sample_size_by_cat*len(self.unique_label) > len(data_out):
            indices = [i for i in range(len(data_out))]
        else :
            # Randomly sample the data
            for label in self.unique_label:
                indices_label = []
                for i in range(len(labels)):
                    if (labels[i] == label):
                        indices_label.append(i)
                n = min(sample_size_by_cat,len(indices_label))
                indices_label = random.sample(indices_label,n)
                indices.extend(indices_label)
        
        sampled_data_out = data_out[indices]
        sampled_labels = labels[indices]
        df = pd.DataFrame(sampled_data_out, columns=features)
        df['Cluster'] = sampled_labels

        pairplot = sns.pairplot(df, hue='Cluster', palette='viridis')
        plt.suptitle('Pairwise Plots of Features with Cluster Labeling (Sampled Data)', y=1.02)

        # Save the plot to a file
        pairplot.savefig(save_path)
        plt.close()
            

        


    