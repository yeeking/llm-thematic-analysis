import ta_utils
import os 
import json
import sys
import pandas as pd 
from tqdm import tqdm
from openai import OpenAI



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


if __name__ == "__main__":
    print("Gettting tags from a collection and summarising")
    print(sys.argv)
    # assert len(sys.argv) == 5, "Usage: python script.py  json_tags_to_quotes_File json_tag_z_score_file json_merged_tag_file csv_codebook"
    assert len(sys.argv) == 3, "Usage: python script.py  json_tags_to_quotes_File llm-model"
    
    # print(sys.argv)
    # collection_name = sys.argv[1]
    json_naive_tag_file = sys.argv[1] # read this in 
    # codebook_csv_file = sys.argv[2] # write this out   
    assert os.path.exists(json_naive_tag_file), f"Cannot find tag input file {json_naive_tag_file}"
    model = sys.argv[2]

    # extract the 'naive' tags from step 2a
    with open(json_naive_tag_file) as f:
        j_in_str = f.read()
    tags_to_quotes = json.loads(j_in_str)
    tags_to_quotes = merge_tags_on_case(tags_to_quotes)
    tag_list = [t for t in tags_to_quotes.keys()]

    print(f"Got {len(tag_list)} tags. Computing embeds ")
    # build a semantic distance matrix between all tags
    # tags_to_embs = compute_tag_embeddings(tag_list)

    # model = "llama3.1:70b"
    # model = "llama3.2:3b-instruct-q8_0"
    # model=  "llama3.1:8b"

    tags_to_embs = {}
    with tqdm(total=len(tag_list), desc="Processing Tags") as pbar:
        # tags_to_embs = compute_tag_embeddings_via_description(tag_list, model)
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

        for t in tag_list:
            if pd.isna(t):
                print(f"Found bad tag. Skipping {t}")
                continue
            # v1 generate a description of the tag and compute the embedding of the 
            # description
            # description, embedding = ta_utils.compute_tag_embeddings_via_description(t, model)
            # v2 alternative - just use the tag
            embedding = ta_utils.text_to_embeddings(t)
            description = ""
            # v3 use lm-studio api
            # description, embedding = ta_utils.compute_tag_embeddings_via_description_lmstudio(client, t)
            # print(f"Tag {t}: Got description: {description}")
            tags_to_embs[t] = {"embedding":embedding, "description":description}
            # Point to the local server
           
            pbar.update(1)  # Update the progress bar by one step

    
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame({
        'tag': list(tags_to_embs.keys()),
        'description': [json.dumps(embedding["description"]) for embedding in tags_to_embs.values()],
        'embeddings': [json.dumps(embedding["embedding"]) for embedding in tags_to_embs.values()]
    })

    # Display the resulting DataFrame
    print(df.head())
    df.to_csv(json_naive_tag_file[0:-5] + "_embeddings.csv")


        