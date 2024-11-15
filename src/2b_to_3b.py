
# # run the python scripts from 2b to 3b an all the models, all the datasets
# import os 
# datasets = ["collusionmac", "examsettermac"]
# models = ["gemma27b", "llama323b", "llama3170b"]

# for dataset in datasets:
#     for model in models:
#         tag_json_file = f"{dataset}{model}.json"
#         assert os.path.exists(tag_json_file), f"Cannot find tag file for {dataset} {model} : {tag_json_file}"
#         runner = f"python 2b_clean_tags.py  {tag_json_file}"
#         clean_tag_csv_file = tag_json_file[0:-5] + "_cleaned.csv"
#         assert os.path.exists(clean_tag_csv_file), f"Cannot find cleaned tag file {clean_tag_csv_file}"
#         runner = f"python 2c_extract_embeddings.py {clean_tag_csv_file}"
#         embeddings_csv_file = clean_tag_csv_file[0:-4] + "_embeddings.csv"
#         assert os.path.exists(embeddings_csv_file), f"Cannot find embeddings file {embeddings_csv_file}"
#         runner = f"python 3a_cluster_analysis.py {embeddings_csv_file} ../plots/"
#         runner = f"python 3b_cluster_labels.py {embeddings_csv_file} ../plots/"
        

import os
import subprocess
import sys

if __name__ == "__main__":

    assert len(sys.argv) == 3, f"Usage python script.py data_folder 2a,2b,2c,3a,3b,4a - you passed {len(sys.argv)} args"
    # datasets = ["collusionmac", "examsettermac"]
    # models = ["gemma27b", "llama323b", "llama3170b"]

    datasets = ["examsetterv2"]
    models = ["gemma27b"]

    data_folder = sys.argv[1]

    assert os.path.exists(data_folder), f"Cannot find data folder {data_folder}"

    try:
        stages = sys.argv[2].split(",")
    except:
        print(f"Stages should be a command separated list of steps, e.g. 2a,2b,2c,3a,3b,4a")
        sys.exit(1)


    for dataset in datasets:
        for model in models:
            tag_json_file = f"{data_folder}{dataset}{model}.json"            
            clean_tag_json_file = tag_json_file[0:-5] + "_cleaned.json"
            clean_tag_csv_file = tag_json_file[0:-5] + "_cleaned.csv"

            embeddings_csv_file = clean_tag_json_file[0:-5] + "_embeddings.csv"
            cluster_csv_file = f"{embeddings_csv_file[0:-4]}_clusters.csv"
            theme_csv_file = f"{embeddings_csv_file[0:-4]}_clusters_themes.csv"
        

            if "2b" in stages:
                # CLEAN TAGS
                assert os.path.exists(tag_json_file), f"Cannot find tag file for {dataset} {model} : {tag_json_file}"
                runner = f"python 2b_clean_tags.py  {tag_json_file}"
                print(f"Running: {runner}")
                result = subprocess.run(runner, shell=True)
                assert result.returncode == 0, f"Script {runner} failed "
            if "2c" in stages:
                # ## EXTRACT EMBEDDINGS
                assert os.path.exists(clean_tag_json_file), f"Cannot find cleaned tag file {clean_tag_json_file}" 
                runner = f"python 2c_tags_to_embeddings.py {clean_tag_json_file} none"
                print(f"Running: {runner}")
                result = subprocess.run(runner, shell=True)
                assert result.returncode == 0, f"Script {runner} failed "
            if "3a" in stages:
                ## GENERATE CLUSTER PLOTS
                # python 3b_cluster_labels.py ../data/collusionmacgemma27b_cleaned_embeddings.csv ../plots/
                assert os.path.exists(embeddings_csv_file), f"Cannot find embeddings file {embeddings_csv_file}"
                runner = f"python 3a_cluster_visualisation.py {embeddings_csv_file} ../plots/ {dataset}_{model}"
                print(f"Running: {runner}")
                result = subprocess.run(runner, shell=True)
                assert result.returncode == 0, f"Script {runner} failed "

            if "3b" in stages: 
                ## DO CLUSTER LABELLING 
                assert os.path.exists(embeddings_csv_file), f"Cannot find embeddings file {embeddings_csv_file}"
                runner = f"python 3b_embeddings_to_clusters.py {embeddings_csv_file}"
                print(f"Running: {runner}")  
                result = subprocess.run(runner, shell=True)
                assert result.returncode == 0, f"Script {runner} failed "
                print("3b complete. ")
            if "4a" in stages:
                ## convert tags in clusters to themes
                assert os.path.exists(cluster_csv_file), f"Cannot find cluster file {cluster_csv_file}"
                assert os.path.exists(clean_tag_csv_file), f"Cannot find clean tag csv file {clean_tag_csv_file}"
                
                runner = f"python 4a_clusters_to_themes.py {cluster_csv_file} {clean_tag_csv_file} llama3.1:70b-instruct-q5_K_M"
                print(f"Running: {runner}")  
                result = subprocess.run(runner, shell=True)
                assert result.returncode == 0, f"Script {runner} failed "
            
            if "4b" in stages:
                print("Generating TAG and THEME plots")
                ## convert tags in clusters to themes
                # assert os.path.exists(cluster_csv_file), f"Cannot find cluster file {cluster_csv_file}"
                assert os.path.exists(theme_csv_file), f"Cannot find theme csv file {theme_csv_file}"
                assert os.path.exists(embeddings_csv_file), f"Cannot find tag embeddings_csv_file csv file {embeddings_csv_file}"
                
                runner = f"python 4b_theme_visualisation.py {embeddings_csv_file} {theme_csv_file} ../plots/{dataset}_{model}_tags_and_themes.pdf '{dataset} {model} tags and themes via t-SNE'"
                print(f"Running: {runner}")  
                result = subprocess.run(runner, shell=True)
                assert result.returncode == 0, f"Script {runner} failed "

            if "4d" in stages:
                print("Generating THEME plots")
                ## convert tags in clusters to themes
                assert os.path.exists(theme_csv_file), f"Cannot find theme file {theme_csv_file}"
                runner = f"python 4d_plot_theme_embeddeings.py {theme_csv_file} ../plots/{dataset}_{model}_themes.pdf '{dataset} {model}_themes T-sNE'"
                print(f"Running: {runner}")                  
                result = subprocess.run(runner, shell=True)
                assert result.returncode == 0, f"Script {runner} failed "

            
