# 4g_dataset_theme_distance_plot.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
import ta_utils  # Assuming ta_utils is the module where text_to_embeddings is defined
import os
import sys

if __name__ == "__main__":
    assert len(sys.argv) == 7, "usage: python script.py dataset1.csv dataset2.csv dataset1-name dataset2-name plot_file threshold"
    ds1_file = sys.argv[1]
    ds2_file = sys.argv[2]
    ds1_name = sys.argv[3]
    ds2_name = sys.argv[4]
    plot_file = sys.argv[5]
    threshold = float(sys.argv[6])

    assert os.path.exists(ds1_file), f"Cannot find csv 1 file {ds1_file}"
    assert os.path.exists(ds2_file), f"Cannot find csv 1 file {ds2_file}"
        
    # read in the two CSV theme files and compile them into a 

    # Load the CSV file
    data1 = pd.read_csv(ds1_file)
    data2 = pd.read_csv(ds2_file)
    
    # Separate data into 'exam' and 'collusion' based on the dataset column
    # exam_data = data1# data[data['dataset'] == 'exam']
    # collusion_data = data2# data[data['dataset'] == 'collusion']

    # Extract themes and create labels in 'dataset-ID' format
    ds1_themes = sorted(data1['theme'].unique().tolist())
    ds2_themes = sorted(data2['theme'].unique().tolist())

    ds1_labels = [ds1_name + str(ind) for ind,theme in enumerate(ds1_themes)]
    ds2_labels = [ds2_name + str(ind) for ind,theme in enumerate(ds2_themes)]
    
                         
    
    # Compute embeddings for each theme
    ds1_embeddings = [ta_utils.text_to_embeddings(theme) for theme in ds1_themes]
    ds2_embeddings = [ta_utils.text_to_embeddings(theme) for theme in ds2_themes]

    # Calculate cosine distances between each exam and collusion theme
    distance_matrix = np.zeros((len(ds1_themes), len(ds2_themes)))
    for i, ds1_embedding in enumerate(ds1_embeddings):
        for j, ds2_embedding in enumerate(ds2_embeddings):
            # Calculate cosine distance
            distance_matrix[i, j] = cosine(ds1_embedding, ds2_embedding)

    # Convert distance matrix to a DataFrame with custom labels
    distance_df = pd.DataFrame(distance_matrix, index=ds1_labels, columns=ds2_labels)

    # Find and print themes with distances below the threshold
    low_distance_pairs = []
    for i, row in distance_df.iterrows():
        for j, value in row.items():
            if value < threshold:
                low_distance_pairs.append((i, j))
                # # Extract detailed info for exam and collusion themes
                # exam_info = exam_data[exam_data['ID'] == int(i.split('-')[1])].iloc[0]
                # collusion_info = collusion_data[collusion_data['ID'] == int(j.split('-')[1])].iloc[0]
                # print(f"Low distance pair: Exam Theme '{exam_info['theme']}' (ID: {exam_info['ID']}, Dataset: {exam_info['dataset']}) "
                #       f"and Collusion Theme '{collusion_info['theme']}' (ID: {collusion_info['ID']}, Dataset: {collusion_info['dataset']}), "
                #       f"Distance: {value}")

    # Plot heatmap with inverted grayscale
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(distance_df, cmap='Greys_r', cbar_kws={'label': 'Cosine Distance'}, 
                     xticklabels=True, yticklabels=True, linewidths=0.5)

    # Add hatching for low-distance pairs
    for (i_label, j_label) in low_distance_pairs:
        i = ds1_labels.index(i_label)
        j = ds2_labels.index(j_label)
        # Add a patterned overlay with hatching
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, hatch='///', edgecolor='white', lw=2))
        # ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color='white', alpha=0.3))
        # ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, hatch='///', edgecolor='white', lw=0))


    # plt.xlabel('Collusion Themes')
    # plt.ylabel('Exam Themes')
    plt.title('Cosine Distance Heatmap between Exam and Collusion Themes')
    print(f"Saving heatmap plot to {plot_file}")
    plt.savefig(plot_file)
    # plt.show()
    # now write the theme lookup index table

    # Extract themes and create labels in 'dataset-ID' format
    # ds1_themes = sorted(data1['theme'].unique().tolist())
    # ds2_themes = sorted(data2['theme'].unique().tolist())
    # ds1_labels = [ds1_name + str(ind) for ind,theme in enumerate(ds1_themes)]
    # ds2_labels = [ds2_name + str(ind) for ind,theme in enumerate(ds2_themes)]
    for ind in range(len(ds1_labels)):
        print(f"{ds1_labels[ind]},{ds1_themes[ind]}")
    for ind in range(len(ds2_labels)):
        print(f"{ds2_labels[ind]},{ds2_themes[ind]}")
        

