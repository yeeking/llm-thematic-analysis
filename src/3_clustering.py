# https://pypi.org/project/s-dbw/
#  S_Dbw (Scatter-Density between-within) index performed
# best in a variety of simulated clustering scenarios
# Y. Liu, Z. Li, H. Xiong, X. Gao and J. Wu, "Understanding of Internal Clustering Validation Measures," 2010 IEEE International Conference on Data Mining, Sydney, NSW, Australia, 2010, pp. 911-916, doi: 10.1109/ICDM.2010.35. keywords: {Indexes;Noise;Clustering algorithms;Noise measurement;Current measurement;Elbow;Economics},
# 

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn import preprocessing
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.metrics import calinski_harabasz_score

import pandas as pd 
import json 

import matplotlib.pyplot as plt
import sys
import os 

def get_tag_to_embeddings(csv_filename, embedding_field="embeddings"):
    """
    returns a dictionary mapping tags to mebeddings
    """
    data = pd.read_csv(csv_filename)
    tag_embeddings = {}

    for ind,row in data.iterrows():
        tag_embeddings[row['tag']] = np.array(json.loads(row['embeddings']))
    return tag_embeddings
    

def get_pca_variances(embeddings):
    """
    returns a dictionary mapping number of PCA components
    to explained variance, e.g. x[100] = 0.95 # 100 components explains 95% of variance
    """
    maxn = len(embeddings)
    pca_components = np.arange(2, maxn, maxn/100) # 10 values from 2 to max components
    
    # Dictionary to store cumulative explained variance for each component count
    explained_variances = {}
    target = 0.95
    # Loop over each number of components
    for n_components in pca_components:
        print(f"Doing PCA with {n_components}")
        # Apply PCA with n_components
        n_components = int(n_components)
        pca = PCA(n_components=n_components)
        # print(embeddings)
        pca.fit(embeddings)  # 'embeddings' should be your dataset of vectors

        # Calculate cumulative explained variance
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)[-1]  # Total explained variance for n_components
        explained_variances[n_components] = cumulative_variance

        if cumulative_variance >= target:
            best_n = n_components
            break
        print(f"PCA components: {n_components}, Cumulative explained variance: {cumulative_variance:.4f}")
    
    return explained_variances, n_components

def plot_pca_results(explained_variances:dict, outfile='pca.png'):
    """
    plots the results received by get_pca_variances
    """
    # Plot cumulative explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(list(explained_variances.keys()), list(explained_variances.values()), marker='o')
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance vs Number of PCA Components')
    plt.grid(True)
    plt.savefig(outfile)
    # plt.show()


def get_cluster_scores(pca_components, embeddings, cluster_range=[2,4,6,8,10,20]):
    """
    returns a dictionary mapping number of clusters -> davies_bouldin and silhouette scores
    """
    # Apply PCA to reduce dimensions
    if pca_components > 0:
        pca = PCA(n_components=pca_components)
        # reduce dimensionality
        reduced_embeddings = pca.fit_transform(embeddings)
        # convert for cosine distance
        normed_embeddings = preprocessing.normalize(reduced_embeddings)
    else: 
        normed_embeddings = preprocessing.normalize(embeddings)

    # Dictionary to store Davies-Bouldin and Silhouette scores for each k
    results = {}

    # Loop over each number of clusters (k)
    for k in cluster_range:
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=k, random_state=0)
        cluster_labels = kmeans.fit_predict(normed_embeddings)

        # Calculate Davies-Bouldin Index
        db_index = davies_bouldin_score(normed_embeddings, cluster_labels)
        
        # Calculate Silhouette Score (only if k > 1)
        silhouette_avg = silhouette_score(normed_embeddings, cluster_labels)

        ch_score = calinski_harabasz_score(normed_embeddings, cluster_labels)
        
        
        results[k] = {'davies_bouldin': db_index, 
                      'silhouette': silhouette_avg, 
                      "Calinski_Harabasz":ch_score}
        
        print(f"PCA Dim: {pca_components}, Clusters: {k}, DB Index: {db_index:.3f}, Silhouette Score: {silhouette_avg:.3f}" if k > 1 else f"PCA Dim: {dim}, Clusters: {k}, DB Index: {db_index:.3f}")

    return results


def plot_cluster_scores(k_to_scores:dict, feature, outfile, pca_components):
    
    x_values = list(k_to_scores.keys()) # k is the thing we vary
    y_values = [d[feature] for d in k_to_scores.values()]
    
    plt.figure(figsize=(12, 8))
    plt.plot(x_values, y_values, marker='o')#, label=f'PCA {dim} dims')
    
    # # Add labels with arrows pointing to the last point in each line
    # plt.annotate(f'PCA {dim} dims',
    #             xy=(x_values[-1], y_values[-1]),               # Last point coordinates
    #             xytext=(x_values[-1] + 0.5, y_values[-1]),     # Offset the text slightly
    #             arrowprops=dict(arrowstyle="->", lw=1.5))

    plt.xlabel('Number of Clusters (k)')
    plt.ylabel(feature)
    plt.title(f'{feature} vs Number of Clusters with {pca_components} dim')
    plt.legend()
    plt.grid(True)
    plt.savefig(outfile)
    # plt.show()


if __name__ == "__main__":
    assert len(sys.argv) == 4, f"USage: python script.py tags_and_embeddings_csv plot_folder run_name"

    csv_file = sys.argv[1]
    plot_folder = sys.argv[2]
    run_name = sys.argv[3] # 
    assert os.path.exists(csv_file)
    assert os.path.exists(plot_folder)

    print(f"Loading data file {csv_file}")

    tags_to_enbs = get_tag_to_embeddings(csv_file)
    embeddings = list(tags_to_enbs.values())
    pca_results, best_n = get_pca_variances(embeddings)
    plot_pca_results(pca_results, plot_folder + "/"+run_name+"_pca.png")
    cluster_scores = get_cluster_scores(best_n, embeddings, cluster_range=range(2, 100))
    # zero for pca_comps forces it to use the complete embedding vector for clustering 
    # cluster_scores = get_cluster_scores(0, embeddings, cluster_range=range(2, 100))
    for ind in cluster_scores.keys():
        features = cluster_scores[ind].keys()
        break
    print(f"Found cluster features {features} for ks {cluster_scores.keys()}")
    for f in features:
        fname = f"{plot_folder}/{run_name}_{f}.png"
        plot_cluster_scores(cluster_scores, f, fname, best_n)





