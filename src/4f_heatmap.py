import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    assert len(sys.argv) == 3, f"Usage python script.py input.csv output_plot_file"
    # Load CSV file
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    data = pd.read_csv(input_file)
    assert "t1" in data.keys(), f"CSV needs a t1 and a t2 row - got {data.keys()}"
    assert "t2" in data.keys(), f"CSV needs a t1 and a t2 row - got {data.keys()}"
    
    # Check for complete matrix
    unique_t1 = data['t1'].unique()
    unique_t2 = data['t2'].unique()

    # Create all possible (t1, t2) pairs, excluding pairs where t1 == t2
    all_pairs = {(t1, t2) for t1 in unique_t1 for t2 in unique_t2 if t1 != t2}
    actual_pairs = set(zip(data['t1'], data['t2']))

    # Assert that all (t1, t2) pairs are present in the data
    missing_pairs = all_pairs - actual_pairs
    assert not missing_pairs, f"Missing pairs in data: {missing_pairs}"

    # Calculate mean of `cosine_z` and `quote_z` for each (t1, t2) pair
    data['mean_z'] = data[['cosine_z', 'quote_z']].mean(axis=1)
    #mean_matrix = data.pivot_table(index='t1', columns='t2', values='mean_z')
    mean_matrix = data.pivot_table(index='t1', columns='t2', values='cosine_z')
    # simpler theme titles for display 
    # theme_mapping = {label: f'theme {i+1}' for i, label in enumerate(sorted(set(unique_t1).union(set(unique_t2))))}
    # # Rename the index and columns in the matrix with "theme" labels
    # mean_matrix.index = mean_matrix.index.map(theme_mapping)
    # mean_matrix.columns = mean_matrix.columns.map(theme_mapping)


    # Plot heatmap
    plt.figure(figsize=(12, 10))
    # sns.heatmap(mean_matrix, annot=False, cmap='coolwarm', cbar_kws={'label': 'Mean Z'})
    # sns.heatmap(mean_matrix, annot=False, cmap='coolwarm_r', cbar_kws={'label': 'Mean Z'})  # '_r' inverts the color map
    sns.heatmap(mean_matrix, annot=False, cmap='Greys_r', cbar_kws={'label': 'Mean Z'})  # 'Greys_r' for inverted grayscale
    # plt.xlabel('t2')
    # plt.ylabel('t1')
    plt.xlabel('t2 (Themes)')
    plt.ylabel('t1 (Themes)')
    plt.title('Heatmap of Mean Z Scores for (t1, t2) Pairs')
    print(f"Saving plot to {output_file}")
    plt.subplots_adjust(left=0.5, right=0.9, top=0.9, bottom=0.5)  # Adjust as needed
    plt.savefig(output_file)
    plt.close()
    for x in [(ind,t) for ind,t in enumerate(unique_t1)]:
        print(x)

    print("Done")
