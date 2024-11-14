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
    assert len(sys.argv) == 4, "usage: python script.py csv_file plot_file threshold"
    csv_file = sys.argv[1]
    plot_file = sys.argv[2]
    threshold = float(sys.argv[3])
    assert os.path.exists(csv_file), f"Cannot find csv file {csv_file}"
    
    # Load the CSV file
    data = pd.read_csv(csv_file)

    # Separate data into 'exam' and 'collusion' based on the dataset column
    exam_data = data[data['dataset'] == 'exam']
    collusion_data = data[data['dataset'] == 'collusion']

    # Extract themes and create labels in 'dataset-ID' format
    exam_themes = exam_data['theme'].tolist()
    collusion_themes = collusion_data['theme'].tolist()
    exam_labels = [f"{row['dataset']}-{row['ID']}" for _, row in exam_data.iterrows()]
    collusion_labels = [f"{row['dataset']}-{row['ID']}" for _, row in collusion_data.iterrows()]

    # Compute embeddings for each theme
    exam_embeddings = [ta_utils.text_to_embeddings(theme) for theme in exam_themes]
    collusion_embeddings = [ta_utils.text_to_embeddings(theme) for theme in collusion_themes]

    # Calculate cosine distances between each exam and collusion theme
    distance_matrix = np.zeros((len(exam_themes), len(collusion_themes)))
    for i, exam_embedding in enumerate(exam_embeddings):
        for j, collusion_embedding in enumerate(collusion_embeddings):
            # Calculate cosine distance
            distance_matrix[i, j] = cosine(exam_embedding, collusion_embedding)

    # Convert distance matrix to a DataFrame with custom labels
    distance_df = pd.DataFrame(distance_matrix, index=exam_labels, columns=collusion_labels)

    # Find and print themes with distances below the threshold
    low_distance_pairs = []
    for i, row in distance_df.iterrows():
        for j, value in row.items():
            if value < threshold:
                low_distance_pairs.append((i, j))
                # Extract detailed info for exam and collusion themes
                exam_info = exam_data[exam_data['ID'] == int(i.split('-')[1])].iloc[0]
                collusion_info = collusion_data[collusion_data['ID'] == int(j.split('-')[1])].iloc[0]
                print(f"Low distance pair: Exam Theme '{exam_info['theme']}' (ID: {exam_info['ID']}, Dataset: {exam_info['dataset']}) "
                      f"and Collusion Theme '{collusion_info['theme']}' (ID: {collusion_info['ID']}, Dataset: {collusion_info['dataset']}), "
                      f"Distance: {value}")

    # Plot heatmap with inverted grayscale
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(distance_df, cmap='Greys_r', cbar_kws={'label': 'Cosine Distance'}, 
                     xticklabels=True, yticklabels=True, linewidths=0.5)

    # Add hatching for low-distance pairs
    for (i_label, j_label) in low_distance_pairs:
        i = exam_labels.index(i_label)
        j = collusion_labels.index(j_label)
        # Add a patterned overlay with hatching
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, hatch='///', edgecolor='white', lw=2))
        # ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color='white', alpha=0.3))
        # ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, hatch='///', edgecolor='white', lw=0))


    # plt.xlabel('Collusion Themes')
    # plt.ylabel('Exam Themes')
    plt.title('Cosine Distance Heatmap between Exam and Collusion Themes')
    plt.savefig(plot_file)
    # plt.show()
