
### Generates a plot with all the themes shown as numerical boxes and all the tags shown as symbols
### each tag has a symbol assigned based on cluster
### This is the one that went into the paper
### under the 

## Plot tags as markers with themes overlaid. -
## This is figure 

import pandas as pd
import numpy as np
import sys
import os 
import ta_utils
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.markers as mmarkers
import textwrap


## First combine the existing data file 
def combine_files(f1, f2):
    # Load the data from the two CSV files
    # df1 = pd.read_csv('collusionmacllama3170b_cleaned_embeddings.csv')
    # df2 = pd.read_csv('collusionmacllama3170b_cleaned_embeddings_clusters_themes.csv')
    df1 = pd.read_csv(f1)
    assert "embeddings" in df1.keys(), f"Embeddings file {f1} does not seem to have embedding key in here: {df1.keys()}"

    df2 = pd.read_csv(f2)
    assert "theme" in df2.keys(), f"Embeddings file {f2} does not seem to have theme key in here: {df1.keys()}"
    
    # Initialize lists to store cluster and theme values
    clusters = []
    themes = []
    quote_counts = []

    
    # Create a dictionary from df2 for quick lookup by tag
    def get_matching_row(df, search_field, search_value):
        res = df[df[search_field] == search_value]
        assert len(res) > 0, f"Could not find {search_field} == {search_value}"
        first_match = df[df[search_field] == search_value].iloc[0]
        return first_match
    # df2_lookup = df2.set_index('tag')[['cluster', 'theme', 'quotes']].to_dict(orient='index')
    

    # Iterate over each row in df1, look up matching cluster and theme from df2
    for tag in df1['tag']:
        # assert tag in df2_lookup, f"{tag} not in set of size {df2_lookup}"
        match_row = get_matching_row(df2, "tag", tag)
        clusters.append(match_row['cluster'])
        themes.append(match_row['theme'])
        quotes = json.loads(match_row['quotes'])
        # print(f"Tag {tag} quuote count {len(quotes)}")
        quote_counts.append(len(quotes))
        # else:
        #     print(f"Weirdness - tag {tag} not  in {df2_lookup.keys()}")
        #     clusters.append(None)  # Optional: Handle cases where tag is not found in df2
        #     themes.append(None)
        #     quote_counts.append(None)

    # assert len(clusters) == len(df1['tag']), f"Problem merging files - did not find a cluster for every tag. Tags: {len(df1['tag'])} clusters {len(clusters)}"
    # assert len(themes) == len(df1['tag']), f"Problem merging files - did not find a theme for every tag. Tags: {len(df1['tag'])} clusters {len(v)}"
    
    # Assign the lists as new columns in df1
    df1['cluster'] = clusters
    df1['theme'] = themes
    df1['quote_count'] = quote_counts

    # Save the updated dataframe to 'tags.csv'
    df1.to_csv('tags.csv', index=False)

def do_tsne_plot_v2(csv_file, title, plot_file, theme_stats_df, label_theme_with_index = True):

    # Load data from CSV
    data = pd.read_csv(csv_file)
    print(f"Loaded data for plotting from {csv_file} Unique clusters: {len(data['cluster'].unique())}")
    # Convert the JSON string in the embeddings field to an actual vector
    data['embedding_vector'] = data['embeddings'].apply(json.loads)

    # make a theme -> index look up based on making alphabetical list of themes
    theme = data["theme"].unique()
    theme_to_ind = ta_utils.get_theme_to_ind_lookup(theme)


    # Stack all embeddings into a 2D numpy array
    embeddings = np.vstack(data['embedding_vector'].values)

    # Reduce dimensions using t-SNE
    print("TSNE fitting")
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    print("Plotting")
    # Add the reduced d imensions to the DataFrame
    data['x'] = reduced_embeddings[:, 0]
    data['y'] = reduced_embeddings[:, 1]

    # Set up markers and grayscale color scheme
    # symbols = list(mmarkers.MarkerStyle.filled_markers)  # Predefined set of marker symbols
    # symbols.extend([u"\u25A0", u"\u25A1", u"\u25B2", u"\u25B3", u"\u25BC", u"\u25BD"])
    # # Start with the predefined markers
    symbols = list(mmarkers.MarkerStyle.filled_markers)

    # Add more unique symbols
    symbols.extend(['+', 'x', 'd', '|', '_', '*', 'h', 'H', '^', 'v', '<', '>', '1', '2', '3', '4'])

    print(f"Available marker {len(symbols)}")
    num_clusters = len(data['cluster'].unique())
    grayscale_colors = [str(i / num_clusters) for i in range(num_clusters)]  # Grayscale colors

    # Helper function to wrap text without breaking words
    def wrap_text(text, width=30):
        return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))

    # Plot
    plt.figure(figsize=(20, 10))
    # plt.figure(figsize=(28, 20))
    # Loop through each unique cluster and plot points with different symbols and grayscale colors
    for idx, (cluster, color, marker) in enumerate(zip(sorted(data['cluster'].unique()), grayscale_colors, symbols)):
        print(f"Cluster {idx}")
        cluster_data = data[data['cluster'] == cluster]
        plt.scatter(
            cluster_data['x'], cluster_data['y'], 
            label=f'Cluster {cluster}', 
            s=50, alpha=0.6, 
            c=color, marker=marker
        )
        
        # Calculate the centroid of the cluster
        centroid_x = cluster_data['x'].mean()
        centroid_y = cluster_data['y'].mean()
        
        # Get the theme name(s) for this cluster, prepend cluster number, and wrap text if it exceeds 30 characters
        theme = ", ".join(cluster_data['theme'].unique())# ?? a bit bonkers... GPT attack
        print(f"Plotting theme {theme} index {theme_to_ind[theme]}")
        if label_theme_with_index:
            theme_text = theme_to_ind[theme]
            fontsize = 20
            # Place the label with theme(s) at the centroid with a semi-transparent box
            stats = theme_stats_df[theme_stats_df["theme"] == theme]

            print(f"Got stats for theme: {list(stats['exam_ratio'])}")

            # theme_shade = list(theme_stats_df[theme_stats_df["theme"] == theme]["collusion_ratio"])[0]
            # theme_shade = np.random.uniform(0, 1)
            theme_shade = list(stats['exam_ratio'])[0]
            # this box is used to show the ratio from exam to collusion of the quotes
            # for this theme
            plt.text(
                centroid_x, centroid_y-3.0, theme_text, 
                fontsize=fontsize, fontweight='bold', ha='center', va='center', color=(theme_shade, theme_shade, theme_shade),
                bbox=dict(facecolor=(theme_shade, theme_shade, theme_shade), alpha=1.0, edgecolor='black', boxstyle='round,pad=0.1')
            )

            plt.text(
                centroid_x, centroid_y, theme_text, 
                fontsize=fontsize, fontweight='bold', ha='center', va='center', color='white',
                bbox=dict(facecolor='black', alpha=1.0, edgecolor='black', boxstyle='round,pad=0.1')
            )
        else: # show the proper title
            theme_text = wrap_text(f"{cluster}: {theme}", width=30)
            fontsize = 10
              # Place the label with theme(s) at the centroid with a semi-transparent box
            plt.text(
                centroid_x, centroid_y, theme_text, 
                fontsize=fontsize, fontweight='bold', ha='center', va='center', color='black',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray', boxstyle='round,pad=0.5')
            )
    fontsize = 20
    plt.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color='white', edgecolor='black', label='Exam', linewidth=1.5),
            plt.Rectangle((0, 0), 1, 1, color='gray', edgecolor='black', label='Both', linewidth=1.5),
            plt.Rectangle((0, 0), 1, 1, color='black', edgecolor='black', label='Collusion', linewidth=1.5)
        ],
        loc='upper right',  # Adjust as needed
        title="Theme mixture",
        fontsize=fontsize,  # Set font size
        facecolor="lightgray",  # Set legend background color to grey
        title_fontsize=fontsize  # Set title font size
    )  
      
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    # plt.title(f"Tag clusters with themes for {title} via 2D t-SNE")
    plt.title(title)
    # plt.legend()
    # plt.show()
    print(f"Saving plot to {plot_file}")
    plt.savefig(plot_file)


if __name__ == "__main__":
    assert len(sys.argv) == 6, f"Usage python script.py embeddings_csv themes.csv theme_stats.csv plot_file plot title"
    f1 = sys.argv[1]
    f2 = sys.argv[2]
    theme_stats_csv = sys.argv[3]
    plot_file = sys.argv[4]
    plot_title = sys.argv[5]
    
    assert os.path.exists(f1), f"Embeddings file not found {f1}"
    assert os.path.exists(f2), f"Themes file not found {f2}"
    assert os.path.exists(theme_stats_csv), f"Themes stats file not found {theme_stats_csv}"
    
    print("Combining data:")
    
    # f1 = 'collusionmacllama3170b_cleaned_embeddings.csv'
    # f2 = 'collusionmacllama3170b_cleaned_embeddings_clusters_themes.csv'
    combine_files(f1, f2)
    theme_stats_data = pd.read_csv(theme_stats_csv)
    want_keys = ["theme_id", "collusion_ratio", "exam_ratio"]
    for k in want_keys: assert k in theme_stats_data.keys(), f"Theme stats missing key {k} - got {theme_stats_data.keys()}"
    
    # title = plot_file[0:-4] # that should have the dataset and model in it 
    print("Doing tSNE and writing plot file")
    do_tsne_plot_v2('tags.csv', plot_title,  plot_file, theme_stats_data)