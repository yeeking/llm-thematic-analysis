import ta_utils
import os 
import json
import sys
import pandas as pd 
from tqdm import tqdm


# def compute_tags_embeddings(tag_list:list):
#     """
#     computes embeddings for short tags - note that this method does 
#     not actually yield good distance metrics 
#     """
#     t_to_embs = {}
#     for i,t in enumerate(tag_list):
#         print(f"Getting embs for {i} of {len(tag_list)}")
#         # t_to_embs[t] = ta_utils.text_to_embeddings(t)
#         t_to_embs[t] = {"embedding":ta_utils.text_to_embeddings(t), "description":t}# changed to make it consistent with the description based one

#     return t_to_embs

# def compute_tags_embeddings_via_description(tag_list:list, model):
#     """
#     This method computes embeddings by first generating a description of the tag
#     suitable to go into the 'code book' which we call 'tag book'. 
#     this module needs to be refactored into 'get description' and compute embeddings
#     """
#     t_to_embs = {}
#     for i,t in enumerate(tag_list):
#         # print(f"Getting embs for {i} of {len(tag_list)}")
#         prompt = f"Please write a description of what you think the following tag is about, given that the tag has been attached to some text from an interview where two people are discussing various topics related to exams in higher education. Tags might refer to the tone of the discussion or the content, or both. Use a neutral tone - just try to guess what the tag relates to. Only provide the tag description please, not your justification of that description, and do not refer to the tag, just provide the description. Here is the tag: '{t}'"
#         description = ta_utils.get_chat_completion(prompt, model)
#         # print(f"First, get a description of tag {t}: \n {description}")
#         t_to_embs[t] = {"embedding":ta_utils.text_to_embeddings(description), "description":description}
#     return t_to_embs


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
    assert len(sys.argv) == 4, "Usage: python script.py  json_tags_to_quotes_File csv_embeddings_file llm-model"
    
    # print(sys.argv)
    # collection_name = sys.argv[1]
    json_naive_tag_file = sys.argv[1] # read this in 
    # json_tag_z_score_file = sys.argv[2] # write this out 
    # json_merged_tag_file = sys.argv[3] # write this out
    codebook_csv_file = sys.argv[4] # write this out 
    

    assert os.path.exists(json_naive_tag_file), f"Cannot find tag input file {json_naive_tag_file}"
        
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
    model = sys.argv[3]
    tags_to_embs = {}
    with tqdm(total=len(tag_list), desc="Processing Tags") as pbar:
        # tags_to_embs = compute_tag_embeddings_via_description(tag_list, model)
        for t in tag_list:
            # generate a description of the tag and compute the embedding of the 
            # description
            # description, embedding = compute_tag_embeddings_via_description(t, model)
            # alternative - just use the tag
            embedding = ta_utils.text_to_embeddings(t)
            description = ""

            tags_to_embs[t] = {"embedding":embedding, "description":description}
            pbar.update(1)  # Update the progress bar by one step

    
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame({
        'tag': list(tags_to_embs.keys()),
        'description': [json.dumps(embedding["description"]) for embedding in tags_to_embs.values()],
        'embeddings': [json.dumps(embedding["embedding"]) for embedding in tags_to_embs.values()]
    })

    # Display the resulting DataFrame
    print(df.head())
    df.to_csv(codebook_csv_file)


        