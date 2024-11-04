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
import ta_utils


def plot_pca_results(explained_variances:dict, outfile='pca.png', run_name=''):
    """
    plots the results received by get_pca_variances
    """
    # Plot cumulative explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(list(explained_variances.keys()), list(explained_variances.values()), marker='o')
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'{run_name}: Cumulative Explained Variance vs Number of PCA Components')
    plt.grid(True)
    plt.savefig(outfile)
    # plt.show()

def plot_cluster_scores(k_to_scores:dict, feature, outfile, pca_components, run_name):
    
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
    plt.title(f'{run_name}:{feature} vs Number of Clusters with {pca_components} dim')
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

    tags_to_enbs = ta_utils.get_tag_to_embeddings(csv_file)
    embeddings = list(tags_to_enbs.values())
    pca_results, best_n = ta_utils.get_pca_variances(embeddings)
    plot_pca_results(pca_results, plot_folder + "/"+run_name+"_pca.png", run_name)
    cluster_scores =ta_utils.get_cluster_scores(best_n, embeddings, cluster_range=range(2, 100))
    # zero for pca_comps forces it to use the complete embedding vector for clustering 
    # cluster_scores = get_cluster_scores(0, embeddings, cluster_range=range(2, 100))
    for ind in cluster_scores.keys():
        features = cluster_scores[ind].keys()
        break
    print(f"Found cluster features {features} for ks {cluster_scores.keys()}")
    for f in features:
        fname = f"{plot_folder}/{run_name}_{f}.png"
        plot_cluster_scores(cluster_scores, f, fname, best_n, run_name)





