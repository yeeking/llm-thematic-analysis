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
    quote_coumts = []

    # Create a dictionary from df2 for quick lookup by tag
    df2_lookup = df2.set_index('tag')[['cluster', 'theme', 'quotes']].to_dict(orient='index')
    

    # Iterate over each row in df1, look up matching cluster and theme from df2
    for tag in df1['tag']:
        if tag in df2_lookup:
            clusters.append(df2_lookup[tag]['cluster'])
            themes.append(df2_lookup[tag]['theme'])
            quotes = json.loads(df2_lookup[tag]['quotes'])
            # print(f"Tag {tag} quuote count {len(quotes)}")
            quote_coumts.append(len(quotes))
        else:
            clusters.append(None)  # Optional: Handle cases where tag is not found in df2
            themes.append(None)
            quotes.append(None)

    # Assign the lists as new columns in df1
    df1['cluster'] = clusters
    df1['theme'] = themes
    df1['quote_count'] = quote_coumts

    # Save the updated dataframe to 'tags.csv'
    df1.to_csv('tags.csv', index=False)

def do_tsne_plot_v1(filename='tags.csv'):

    # Load data from CSV
    data = pd.read_csv(filename)

    # Convert the JSON string in the embeddings field to an actual vector
    data['embedding_vector'] = data['embeddings'].apply(json.loads)

    # Stack all embeddings into a 2D numpy array
    embeddings = np.vstack(data['embedding_vector'].values)

    # Reduce dimensions using t-SNE
    print("TSNE fitting... ")
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    print("Plotting...")
    
    # Add the reduced dimensions to the DataFrame
    data['x'] = reduced_embeddings[:, 0]
    data['y'] = reduced_embeddings[:, 1]

    # Set up markers and grayscale color scheme
    symbols = mmarkers.MarkerStyle.filled_markers  # Predefined set of marker symbols
    num_clusters = len(data['cluster'].unique())
    grayscale_colors = [str(i / num_clusters) for i in range(num_clusters)]  # Grayscale colors

    # Plot
    plt.figure(figsize=(10, 8))

    for idx, (cluster, color, marker) in enumerate(zip(sorted(data['cluster'].unique()), grayscale_colors, symbols)):
        cluster_data = data[data['cluster'] == cluster]
        plt.scatter(
            cluster_data['x'], cluster_data['y'], 
            label=f'Cluster {cluster}', 
            s=50, alpha=0.6, 
            c=color, marker=marker
        )

    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title("t-SNE Scatter Plot of Tags by Cluster (Grayscale)")
    plt.legend()
    plt.show()

def do_tsne_plot_v2(csv_file='tags.csv', plot_file="plot.pdf"):

    # Load data from CSV
    data = pd.read_csv(csv_file)

    # Convert the JSON string in the embeddings field to an actual vector
    data['embedding_vector'] = data['embeddings'].apply(json.loads)

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
    symbols = mmarkers.MarkerStyle.filled_markers  # Predefined set of marker symbols
    num_clusters = len(data['cluster'].unique())
    grayscale_colors = [str(i / num_clusters) for i in range(num_clusters)]  # Grayscale colors

    # Helper function to wrap text without breaking words
    def wrap_text(text, width=30):
        return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))

    # Plot
    # plt.figure(figsize=(10, 8))
    plt.figure(figsize=(28, 20))
    # Loop through each unique cluster and plot points with different symbols and grayscale colors
    for idx, (cluster, color, marker) in enumerate(zip(sorted(data['cluster'].unique()), grayscale_colors, symbols)):
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
        themes = ", ".join(cluster_data['theme'].unique())
        theme_text = wrap_text(f"{cluster}: {themes}", width=30)
        
        # Place the label with theme(s) at the centroid with a semi-transparent box
        plt.text(
            centroid_x, centroid_y, theme_text, 
            fontsize=10, fontweight='bold', ha='center', va='center', color='black',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray', boxstyle='round,pad=0.5')
        )

    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title("t-SNE Scatter Plot of Tags by Cluster (Grayscale with Cluster Labels)")
    plt.legend()
    # plt.show()
    plt.savefig(plot_file)


def do_tsne_plot_v3(csv_file='tags.csv', plot_file="plot.pdf"):
    # Load data from CSV
    data = pd.read_csv(csv_file)

    # Convert the JSON string in the embeddings field to an actual vector
    data['embedding_vector'] = data['embeddings'].apply(json.loads)

    # Stack all embeddings into a 2D numpy array
    embeddings = np.vstack(data['embedding_vector'].values)

    # Reduce dimensions using t-SNE
    print("TSNE fitting")
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    print("Plotting")
    # Add the reduced dimensions to the DataFrame
    data['x'] = reduced_embeddings[:, 0]
    data['y'] = reduced_embeddings[:, 1]

    # Set up markers and grayscale color scheme
    symbols = mmarkers.MarkerStyle.filled_markers  # Predefined set of marker symbols
    num_clusters = len(data['cluster'].unique())
    grayscale_colors = [str(i / num_clusters) for i in range(num_clusters)]  # Grayscale colors

    # Helper function to wrap text without breaking words
    def wrap_text(text, width=30):
        return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))

    # Plot with 16:10 aspect ratio
    plt.figure(figsize=(16, 10))

    # Loop through each unique cluster and plot points with different symbols and grayscale colors
    for idx, (cluster, color, marker) in enumerate(zip(sorted(data['cluster'].unique()), grayscale_colors, symbols)):
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
        
        # Get the theme name(s) for this cluster
        unique_themes = cluster_data['theme'].unique()
        
        theme_text = ""
        
        for theme in unique_themes:
            theme_data = cluster_data[cluster_data['theme'] == theme]
            
            # Calculate the total quote count and number of tags for this theme
            total_quote_count = theme_data['quote_count'].sum()
            tag_count = theme_data.shape[0]
            
            # Format the theme title with the total quote count and tag count
            theme_text += f"{cluster}: {theme} (Quotes: {total_quote_count}, Tags: {tag_count})\n"
        
        # Wrap the theme text to avoid long lines
        theme_text = wrap_text(theme_text.strip(), width=30)
        
        # Place the label with theme(s) at the centroid with a semi-transparent box
        plt.text(
            centroid_x, centroid_y, theme_text, 
            fontsize=10, fontweight='bold', ha='center', va='center', color='black',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray', boxstyle='round,pad=0.5')
        )

    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title("t-SNE Scatter Plot of Tags by Cluster (Grayscale with Cluster Labels)")
    plt.legend()
    plt.savefig(plot_file)

if __name__ == "__main__":
    assert len(sys.argv) == 4, f"Usage python script.py embeddings_csv themes.csv plot_file"
    f1 = sys.argv[1]
    f2 = sys.argv[2]
    plot_file = sys.argv[3]
    
    assert os.path.exists(f1), f"Embeddings file not found {f1}"
    assert os.path.exists(f2), f"Themes file not found {f2}"
    
    print("Combining data:")
    
    # f1 = 'collusionmacllama3170b_cleaned_embeddings.csv'
    # f2 = 'collusionmacllama3170b_cleaned_embeddings_clusters_themes.csv'
    combine_files(f1, f2)
    print("Doing tSNE and writing plot file")
    do_tsne_plot_v2('tags.csv', plot_file)