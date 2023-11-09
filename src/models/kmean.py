import pickle
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix



def picke_load(file_name):
    with open(file_name, 'rb') as file:
        array2d = pickle.load(file)
    return array2d




pickel_f = picke_load("../../data/output/parsing/sparse.pkl")
sparse_matrix = pickel_f[0]
label = pickel_f[1]


# Cluster the sparse matrix using KMeans
kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto").fit(sparse_matrix)

print(len(kmeans.labels_))

print(kmeans.labels_)
