# 4d_plot_theme_embeddeings.py
### Render the themes via embeddings
### and T-sNE
import pandas as pd
import ta_utils
import numpy as np
import sys
import os 
import ta_utils
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.markers as mmarkers
import textwrap

def do_tsne_plot_v2(embeddings, themes, title, plot_file, theme_index = True):
    # Fit t-SNE
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    theme_to_ind = ta_utils.get_theme_to_ind_lookup(themes)

    # Wrap text function
    def wrap_text(text, width=30):
        return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], color="blue", alpha=0.0)

    for i, (x, y) in enumerate(reduced_embeddings):
        # split_ratio = theme_splits[i]
        split_ratio = np.random.random()
        theme_text = theme_to_ind[themes[i]] if theme_index else wrap_text(themes[i], width=30)
        theme_text = str(theme_text)
        # Calculate text split
        split_index = int(len(theme_text) * split_ratio)
        text_black = theme_text[:split_index]
        text_white = theme_text[split_index:]

        # Render text with gradient effect
        # plt.text(
        #     x, y, text_black, 
        #     fontsize=20, fontweight='bold', ha='center', va='center', color='black',
        #     bbox=dict(facecolor='white', alpha=0.3, edgecolor='gray', boxstyle='round,pad=0.5')
        # )
        # plt.text(
        #     x, y, text_white,
        #     fontsize=20, fontweight='bold', ha='center', va='center', color='white',
        #     bbox=dict(facecolor='black', alpha=0.3, edgecolor='gray', boxstyle='round,pad=0.5'),
        #     zorder=10  # Ensure it overlays the black text correctly
        # )


        if theme_index:
            theme_text = theme_to_ind[themes[i]]
            plt.text(
                x, y, theme_text, 
                fontsize=20, fontweight='bold', ha='center', va='center', color='white',
                bbox=dict(facecolor='black', alpha=0.3, edgecolor='gray', boxstyle='round,pad=0.5')
            )
  
        else: 
            theme_text = wrap_text(themes[i], width=30)  # Wrap theme text
            plt.text(
                x, y, theme_text, 
                fontsize=6, fontweight='bold', ha='center', va='center', color='black',
                bbox=dict(facecolor='white', alpha=0.3, edgecolor='gray', boxstyle='round,pad=0.5')
            )
  

    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    # plt.title("t-SNE Plot of Themes")
    plt.title(title)
    plt.savefig(plot_file)

if __name__ == "__main__":
    assert len(sys.argv) == 4, f"Usage python script.py themes_csv plot_file plot title"
    themes_csv = sys.argv[1]
    plot_file = sys.argv[2]
    plot_title = sys.argv[3]

    assert os.path.exists(themes_csv), f"Cannot find csv input file {themes_csv}"

    data = pd.read_csv(themes_csv)
    assert "theme" in data.keys(), f"Cannot find 'theme' field in datafrome keys: {data.keys()}"

    themes = list(data["theme"].unique())
    embeddings = np.array([ta_utils.text_to_embeddings(t) for t in themes])
    print(type(embeddings[0]))
    do_tsne_plot_v2(embeddings, themes, plot_title, plot_file)


