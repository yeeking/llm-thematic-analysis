# 4e_quote_promiscuity.py
### computes quote promiscuity 
### i.e. for a pair of themes, how many quotes are shared?

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
from scipy.spatial import distance


import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


def count_matches(list1, list2):
    matches = 0
    for i1 in list1:
        if i1 in list2: matches += 1
    return matches

if __name__ == "__main__":
    assert len(sys.argv) == 4, f"Usage python script.py themes_csv plot_file plot title"
    themes_csv = sys.argv[1]
    plot_file = sys.argv[2]
    plot_title = sys.argv[3]

    assert os.path.exists(themes_csv), f"Cannot find csv input file {themes_csv}"

    data = pd.read_csv(themes_csv)
    assert "theme" in data.keys(), f"Cannot find 'theme' field in datafrome keys: {data.keys()}"
    print("Computing embeddings")
    themes = list(data["theme"].unique()) # first of all the themes
    embeddings = np.array([ta_utils.text_to_embeddings(t) for t in themes])

    # the data frame has a row for each tag
    # in that row, there is a theme title and a json array of quotes for that tag
    # This code puts the quotes for each theme together as a unique set
    print("Assigning quotes to themes")
    # now collate quotes by theme
    theme_to_quotes = {}
    for theme in themes:
        matching_rows = data[data["theme"] == theme] # rows for this theme
        if theme not in theme_to_quotes:
            theme_to_quotes[theme] = []
        # collect the quotes together for this theme
        for ind,row in matching_rows.iterrows():
            qs = json.loads(row["quotes"]) # quotes for this theme
            for q in qs: 
                if q not in theme_to_quotes[theme]:
                    theme_to_quotes[theme].append(q)
        # 
        # print(f"Theme {theme} has {len(theme_to_quotes[theme])} quotes")


    print(f"Computing theme-theme distances and quote matches")
    theme_stats = {}


    for ind1,theme1 in enumerate(theme_to_quotes.keys()):
        quotes1 = theme_to_quotes[theme1] 
        emb1 = embeddings[ind1]
        # now check if any of these quotes appear in any other theme
        for ind2,theme2 in enumerate(theme_to_quotes.keys()):
            if theme1 == theme2: continue
            key = theme1 + "_" + theme2
            alt_key = theme2 + "_" + theme1
            if key in theme_stats.keys():continue
            if alt_key in theme_stats.keys():continue
            
            quotes2 = theme_to_quotes[theme2]
            emb2 = embeddings[ind2]
            matches = count_matches(quotes1, quotes2)
            dist = distance.cosine(emb1, emb2)
   
            theme_stats[key] = {"dist":dist, "quote_matches":matches}
            # print(theme_stats[key])

 
    # Extract data for plotting
    dists = np.array([theme_stats[key]["dist"] for key in theme_stats])
    quote_matches = np.array([theme_stats[key]["quote_matches"] for key in theme_stats])


    # Compute correlation coefficient and p-value (confidence level)
    correlation, p_value = pearsonr(dists, quote_matches)

    # Fit a linear regression model for the best fit line
    model = LinearRegression()
    dists_reshaped = dists.reshape(-1, 1)
    model.fit(dists_reshaped, quote_matches)
    best_fit_line = model.predict(dists_reshaped)

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.scatter(dists, quote_matches, color="blue", alpha=0.6, label="Data Points")
    plt.plot(dists, best_fit_line, color="red", label="Best Fit Line")

    # Labels, title, and legend
    plt.xlabel("Dist")
    plt.ylabel("Quote Matches")
    plt.title(plot_title)
    text_label = f"Correlation: {correlation:.4f}\nP-value (Confidence): {p_value:.4f}"
    plt.text(max(dists)*0.5, max(quote_matches) * 0.5, text_label, fontsize=10,
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    # plt.title("Scatter Plot of Dist vs. Quote Matches with Best Fit Line")
    plt.legend()
    plt.savefig(plot_file)

    # Print correlation coefficient and confidence level
    print(f"Correlation Coefficient: {correlation:.4f}")
    print(f"P-value (Confidence Level): {p_value:.4f}")

    dist_z = ta_utils.compute_z_scores(dists)
    quote_z = ta_utils.compute_z_scores(quote_matches)
    merges = []
    for i in range(len(dist_z)):
        if (dist_z[i]) < -2 and (quote_z[i] > 2):
            print(f"Merge: {list(theme_stats.keys())[i]}")
            merges.append(1)
        else: merges.append(0)

    quote_dist_data = pd.DataFrame({"key":list(theme_stats.keys()), 
                                    "cosine_dist":dists, 
                                    "cosine_z":dist_z, 
                                    "matched_quotes":quote_matches, 
                                    "quote_z":quote_z, 
                                    "merge":merges})
    out_file = plot_file[:-4] + "_results.csv"
    print("writing results to ", out_file)
    quote_dist_data.to_csv(out_file)

    
    
