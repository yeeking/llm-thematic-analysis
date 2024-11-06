# 4a_clusters_to_themes\
# loads in a csv with tag and cluster columns
# collects tags with same cluster 
# sends collections to LLM and requests a theme label 

import numpy as np
import sys
import os 
import ta_utils
import pandas as pd 


if __name__ == "__main__":
    assert len(sys.argv) == 3, f"USage: python script.py tags_and_clusters.csv tags_and_quotes.csv"
    clusters_csv = sys.argv[1]
    quotes_csv = sys.argv[2]
    assert os.path.exists(clusters_csv), f"Cannot find clusters csv file {clusters_csv}"
    assert os.path.exists(quotes_csv), f"Cannot find clusters csv file {quotes_csv}"

    data = pd.read_csv(clusters_csv)
    quote_data = pd.read_csv(quotes_csv)
    # get clustdr -> tag dict 
    clusters_to_tags = {}
    for ind,row in data.iterrows():
        tag = row["tag"]
        cluster = row["cluster"]
        if cluster not in clusters_to_tags.keys():
            clusters_to_tags[cluster] = []
        clusters_to_tags[cluster].append(tag)
    theme_dict = {}
    for c_ind in clusters_to_tags.keys():
        tag_set = ",".join(clusters_to_tags[c_ind])
        print(f"Cluster {c_ind} has {len(clusters_to_tags[c_ind])} tags.")
        prompt = f"I have collected the following phrases together because they relate to a theme. Please can you write a descriptive title for the theme that represents the important elements of the phrases. Here are the phrases: \"{tag_set}\". You only need to print out the theme title. Do not explain the title. "
        theme_title = ta_utils.get_chat_completion_lmstudio(prompt, "model") # can't dynamically load models anyways
        # theme_title = f"test {c_ind}"
        theme_dict[c_ind] = theme_title
        print(f"Got theme title {theme_title} for tags {tag_set[0:2]}")
    
    # add the theme titles to the data frame 
    # also add quotes back in for theme checking 
    # and write it back out again 
    theme_titles = []
    tag_quotes = []
    for ind,row in data.iterrows():
        c_ind = row["cluster"]
        tag = row["tag"]
        # get the quotes from the other data frame, but check first :) 
        # print(f"Doing lookup for quotes on {tag}. Quote data keys {quote_data.keys()}")
        q_rows = quote_data[ quote_data["tag"] == tag]
        assert len(q_rows) == 1, f"Error finding quotes for tag {tag}"
        this_tags_quotes = q_rows["quotes"].iloc[0]
        tag_quotes.append(this_tags_quotes)
        theme_titles.append(theme_dict[c_ind])
    data["theme"] = theme_titles
    data["quotes"] = tag_quotes
    
    outfile = clusters_csv[0:-4] + "_themes.csv"
    print(f"Writing themes to {outfile}")
    data.to_csv(outfile)


    
    # instruct it to combine similar themes? 
    
    
