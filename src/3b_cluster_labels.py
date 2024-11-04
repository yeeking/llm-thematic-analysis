
## This script composes the tags assigned in step 2 
## into themes. 
## essentially it carries out a clustering process
## on the tags seeking optimal number of clusters
## based on the classic methods of centroid distance and low overlap

# a) cluster the tags by cosine distance, optimising for 
# spread and overlap as is the standard method for clustering 
# b) According to Braun this process is focused on the codes 
# as opposed to the extracts, i.e. we are not digging back into 
# the text extracts attached to the codes here 
# c) Can also combine codes into sub themes here and REJECT unwanted codes 

import numpy as np
import sys
import os 
import ta_utils
import pandas as pd 


if __name__ == "__main__":
    assert len(sys.argv) == 3, f"USage: python script.py tags_and_embeddings_csv run_name"

    csv_file = sys.argv[1]
    assert os.path.exists(csv_file), f"Cannot find csv embeddings file {csv_file}"
    run_name = sys.argv[2]

    print(f"Loading {csv_file}")

    # first decide the optimal pca count is
    tags_to_enbs = ta_utils.get_tag_to_embeddings(csv_file)
    embeddings = list(tags_to_enbs.values())

    print(f"Computing optimal pca components")
    pca_results, best_pca_n = ta_utils.get_pca_variances(embeddings)
    print(f"Optimal pca components {best_pca_n}")

    print(f"Getting optimal cluster count. ")
    
    cluster_scores = ta_utils.get_cluster_scores(best_pca_n, embeddings, cluster_range=range(2, 100))
    feature = "silhouette"
    feature_scores=  [v[feature] for v in cluster_scores.values()]

    silhouette_target = 0.8 # select cluster count that is 80% of highest score 
    min_score = np.min(feature_scores)
    max_score = np.max(feature_scores)
    threshold_score = silhouette_target * max_score
    print(f"Sihouette score in range {min_score} to {max_score} - target {threshold_score}")
    # now find the k that goes over the targer
    for k,i in enumerate(cluster_scores.keys()):
        if feature_scores[i] >= threshold_score:
            print(f"Went over {silhouette_target} with score of {threshold_score} at k {k}")
            best_k = k
            break
    
    print(f"Proceeding with pca {best_pca_n} and k {best_k}")
    labels = ta_utils.cluster_items(embeddings, pca_n=best_pca_n, k=best_k)
    data = pd.read_csv(csv_file)
    data["embeddings"] = ["" for ind,row in data.iterrows()]
    data["cluster"] = [labels[ind] for ind,row in data.iterrows()]
    outfile = csv_file[0:-4] + "_clusters.csv"
    print(f"Saving cluster labels to {outfile}")
    data.to_csv(csv_file[0:-4] + "_clusters.csv")
    
    
    


