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

def do_tsne_plot_v4(embeddings, themes, title, plot_file):
    # Fit t-SNE
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Wrap text function
    def wrap_text(text, width=30):
        return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))

    # Plotting
    plt.figure(figsize=(12, 8))  # Adjust figure width to accommodate legend
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], color="blue", alpha=0.5)

    # Numerical labels at each (x, y) point
    for i, (x, y) in enumerate(reduced_embeddings):
        plt.text(
            x, y, str(i),  # Display index as label
            fontsize=10, fontweight='bold', ha='center', va='center', color='black',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray', boxstyle='round,pad=0.5')
        )

    # Add a legend with wrapped theme text
    wrapped_themes = [wrap_text(theme, width=30) for theme in themes]
    for i, theme_text in enumerate(wrapped_themes):
        plt.plot([], [], ' ', label=f"{i}: {theme_text}")  # Invisible point with label

    # Position the legend outside the plot area to the right
    plt.legend(
        loc='center left', bbox_to_anchor=(1.05, 0.5), title="Theme Key",
        fontsize=6, title_fontsize='8', frameon=True, edgecolor="gray"
    )

    # Labels and title
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title(title)

    # Adjust layout to fit legend without overlapping
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Reserve space for legend on the right
    plt.savefig(plot_file, bbox_inches="tight")

def do_tsne_plot_v3(embeddings, themes, title, plot_file):
    # Fit t-SNE
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Wrap text function
    def wrap_text(text, width=30):
        return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], color="blue", alpha=0.5)

    # Numerical labels at each (x, y) point and theme legend
    for i, (x, y) in enumerate(reduced_embeddings):
        plt.text(
            x, y, str(i),  # Display index as label
            fontsize=10, fontweight='bold', ha='center', va='center', color='black',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray', boxstyle='round,pad=0.5')
        )

    # Add a legend with wrapped theme text
    wrapped_themes = [wrap_text(theme, width=30) for theme in themes]
    for i, theme_text in enumerate(wrapped_themes):
        plt.plot([], [], ' ', label=f"{i}: {theme_text}")  # Invisible point with label

    plt.legend(loc='upper right', title="Theme Key", fontsize=8, title_fontsize='10', frameon=True, edgecolor="gray")

    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title(title)
    plt.savefig(plot_file)

def do_tsne_plot_v2(embeddings, themes, title, plot_file):
    # Fit t-SNE
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Wrap text function
    def wrap_text(text, width=30):
        return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], color="blue", alpha=0.0)

    for i, (x, y) in enumerate(reduced_embeddings):
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


