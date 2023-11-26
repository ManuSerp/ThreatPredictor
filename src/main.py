"""Main code"""
import numpy as np
from models.spectral_clustering import SpectralClusterModel
from models.kmean import KMeanModel
from models.dbscan import DBSCANModel
from parser.parser import Parser
from parser.feature_reduction import FeatureReduction
from lib.cluster_label import ClusterLabel


# ANSI escape codes for colors
RED = '\033[91m'
GREEN = '\033[92m'
ENDC = '\033[0m' 

def reduction(sparse_array, n_components_ratio = None):
    feature_reduction = FeatureReduction('TruncatedSVD')
    if n_components_ratio is None:
        n_components_ratio = feature_reduction.plot_variance(sparse_array, sparse_array.shape[1], 0.95, v=False)
    feature_reduction.create_model(n_components_ratio)
    data = feature_reduction.fit_transform(sparse_array)

    feature_names = ["duration", "protocol_type0", "protocol_type1", "service0", "service1", "service2", "service3", "service4", "service5", "service6", "flag0", "flag1", "flag2", "flag3", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate"]
    components = feature_reduction.model.get_feature_names_out(feature_names)
    # Getting top features for each component
    top_features = [feature_names[j] for j in components.argsort()[::-1]]

    return data,n_components_ratio, top_features


def main():
    # Write data variable
    PATH_SAVE="../data/output/temp/"
    LOG_PATH="../data/output/parsing/"

    
    parser = Parser(PATH_SAVE,LOG_PATH)
    cluster_model = KMeanModel(n_clusters=100, random_state=0)

    # Load and preprocess data
    s_train,s_test=parser.split_dataset('../data/kddcup.data_10_percent',200000,0.8, encoding="ohe")

    
    # test that the label are the same for train and test set.
    test_label=np.unique(s_test[1])
    try: # lest chech that test label is superior to 5
        assert len(test_label)>5
        print(GREEN + "DATASET TEST PASSED" + ENDC)
    except AssertionError:
        print(RED + "!!DANGEROUS WARNING!!" + ENDC)
        print("The label are not the same for train and test set.")
        print("-------------------------")
        print("train label:")
        print(np.unique(s_train[1]))
        print("test label:")
        print(np.unique(s_test[1]))
        print("-------------------------")
    # Feature reduction
    data_train,n,features=reduction(s_train[0])
    data_test,n,features=reduction(s_test[0],n)
    label_train=s_train[1]
    label_test=s_test[1]

    # Train the cluster model
    # data=sparse_array # remove for feature reduction
    data_out = cluster_model.train(data_train)
    predict_labels_train = cluster_model.predict(data_train)

    cluster_model.evaluating_clustering_performance(data_out,predict_labels_train)

    # Cluster label
    print("==== Cluster label begining ====")
    cluster_label = ClusterLabel(predict_labels_train,label_train,cluster_model.n_clusters)
    cluster_label.cluster_stat()
    predict_label_test=cluster_model.predict(data_test)
    pl=cluster_label.get_predicted_label(predict_label_test)
    res_accuracy=cluster_label.calc_stat(label_test,pl)
    res_metrics=cluster_label.calc_metrics(label_test,pl)
    cluster_label.plot_metrics(res_metrics,res_accuracy)
    cluster_label.plot_combined_pairwise(data_train,label_train,features,20)

if __name__ == '__main__':
    main()





