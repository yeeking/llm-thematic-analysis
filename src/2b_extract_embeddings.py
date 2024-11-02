import ta_utils
import os 
import json
import sys
import pandas as pd 
from tqdm import tqdm


def merge_tags_on_case(tags_and_quotes:dict):
    """
    merges tags that are the same, but with different letter cases
    """
    merged_tags = {}
    for tag1 in tags_and_quotes.keys():
        tag1_low = tag1.lower()
        # add this one to the merged tags
        merged_tags[tag1_low] = tags_and_quotes[tag1]
        for tag2 in tags_and_quotes.keys():
            if tag1 == tag2: continue # I think we skip exact matches
            tag2_low = tag2.lower()
            if tag1_low == tag2_low:# merge tag2_low's quotes into tag1_low's quotes
                merged_tags[tag1_low].extend(tags_and_quotes[tag2])
    print(f"Merged tags went from {len(tags_and_quotes.keys())} to {len(merged_tags.keys())}")
    return merged_tags

def compute_tag_embeddings(tag:str):
    """
    computes embeddings for short tags - note that this method does 
    not actually yield good distance metrics 
    """

    embs = ta_utils.text_to_embeddings(tag) 
    return embs

if __name__ == "__main__":
    print("Gettting tags from a collection and summarising")
    print(sys.argv)
    # assert len(sys.argv) == 5, "Usage: python script.py  json_tags_to_quotes_File json_tag_z_score_file json_merged_tag_file csv_codebook"
    assert len(sys.argv) == 2, "Usage: python script.py  json_tags_to_quotes_File"
    
    # print(sys.argv)
    # collection_name = sys.argv[1]
    json_naive_tag_file = sys.argv[1] # read this in 
    # json_tag_z_score_file = sys.argv[2] # write this out 
    # json_merged_tag_file = sys.argv[3] # write this out
    assert os.path.exists(json_naive_tag_file), f"Cannot find tag input file {json_naive_tag_file}"

    codebook_csv_file = json_naive_tag_file[0:-5] + "-tagonly-codebook.csv" # write this out 
    

        
    # extract the 'naive' tags from step 2a
    with open(json_naive_tag_file) as f:
        j_in_str = f.read()
    tags_to_quotes = json.loads(j_in_str)
    tags_to_quotes = merge_tags_on_case(tags_to_quotes)
    tag_list = [t for t in tags_to_quotes.keys()]

    print(f"Got {len(tag_list)} tags. Computing embeds ")
    # build a semantic distance matrix between all tags
    # tags_to_embs = compute_tag_embeddings(tag_list)

    tags_to_embs = {}
    with tqdm(total=len(tag_list), desc="Processing Tags") as pbar:
        # tags_to_embs = compute_tag_embeddings_via_description(tag_list, model)
        for t in tag_list:
            embedding = compute_tag_embeddings(t)
            tags_to_embs[t] = {"embedding":embedding, "description":""}
            pbar.update(1)  # Update the progress bar by one step

    
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame({
        'tag': list(tags_to_embs.keys()),
        'description': [json.dumps(embedding["description"]) for embedding in tags_to_embs.values()],
        'embeddings': [json.dumps(embedding["embedding"]) for embedding in tags_to_embs.values()]
    })

    # Display the resulting DataFrame
    print(df.head())
    print(f"Saving codebook to {codebook_csv_file}")
    df.to_csv(codebook_csv_file)


        