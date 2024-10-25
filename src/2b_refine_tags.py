### This script takes a set of tags and refines them
### by reducing repetition based on semantic similarity
### i.e. if two tags are excessively semantically similar, 
### they are merged into a single tag
### The definition of 'excessively semantically similar' 
### is based on the distribution of similarities in the tag dataset
### so if the semantic distance between two tags is statistically
### significantly low (z-score) then they are merged 

import sys
import os 
import json
import ta_utils
from scipy.spatial import distance
import pandas as pd 
import numpy as np 

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

def compute_tag_embeddings(tag_list:list):
    """
    computes embeddings for short tags - note that this method does 
    not actually yield good distance metrics 
    """
    t_to_embs = {}
    for i,t in enumerate(tag_list):
        print(f"Getting embs for {i} of {len(tag_list)}")
        t_to_embs[t] = ta_utils.text_to_embeddings(t)
    return t_to_embs

def compute_tag_embeddings_via_description(tag_list:list):
    """
    This method computes embeddings by first generating a description of the tag
    suitable to go into the 'code book' which we call 'tag book'. 
    this module needs to be refactored into 'get description' and compute embeddings
    """
    t_to_embs = {}
    for i,t in enumerate(tag_list):
        print(f"Getting embs for {i} of {len(tag_list)}")
        prompt = f"Please write a description of what you think the following tag is about. Just provide the description please, not your justification of that description. Here is the tag: '{t}'"
        description = ta_utils.get_chat_completion(prompt)
        print(f"First, get a description of tag {t}: \n {description}")
        t_to_embs[t] = {"embedding":ta_utils.text_to_embeddings(description), "description":description}
    return t_to_embs


def compute_tag_distances(tag_list:list, tags_to_embeddings:dict):
    """
    given a list of tags, compute distances between each tag and every other tag
    returns a dict mapping tag1_tag2 keys to distances
    """
    assert type(tag_list) is list, "tag_list should be a list of tags"
    assert type(tags_to_embeddings[list(tags_to_embeddings.keys())[0]]) is list, "tags_to_embeddings should map tags to vector embeddings"
    tag_dists = {}
    for i,t1 in enumerate(tag_list):
        print(f"Computing distances for {i} of {len(tag_list)}")
        for t2 in tag_list:
            k = t1 + "_" + t2
            k2 = t2 + "_" + t1    
            if t1 != t2:
                if k2 in tag_dists:
                    tag_dists[k] = tag_dists[k2]
                else:
                    tag_dists[k] = distance.cosine(tags_to_embeddings[t1], tags_to_embeddings[t2])
    return tag_dists

def compute_z_scores(tag_dists:dict):
    """
    compute the z scores for the distances in the dict 
    returns a dict mapping tag1_tag2 : [distance, z_score]
    """
    assert type(tag_dists[list(tag_dists.keys())[0]]) is np.float64, f"tag_dists should map tags to floats which are the distances between those tags but it is { type(tag_dists[list(tag_dists.keys())[0]])}"
    raw_distances = [tag_dists[k] for k in tag_dists.keys()]
    z_scores = ta_utils.get_z_scores(raw_distances)
    tag_z_scores = {}
    for i,k in enumerate(tag_dists.keys()):
        tag_z_scores[k] = {"distance":raw_distances[i], "z_score":z_scores[i]}
    return tag_z_scores


def remove_repeated_tags(tag_pair_list:list):
    """
    if tag1_tag2 and tag2_tag1 are both in the list, remove one of them
    """
    assert type(tag_pair_list) == list, "tag_pair_list argument should be a list of tags"
    unique_pairs = set()# used to check if paris already added
    filtered_tag_pairs = []
    for pair in tag_pair_list:
        # Split the tags
        tag1, tag2 = pair.split('_')
        # Create the normalized form (sorted tags)
        normalized_pair = f"{min(tag1, tag2)}_{max(tag1, tag2)}"
            # Check if the normalized pair is already in the set
        if normalized_pair not in unique_pairs:
            unique_pairs.add(normalized_pair)
            filtered_tag_pairs.append(pair)
    return filtered_tag_pairs


def semantic_tag_merge(tag_z_scores:dict, tags_and_quotes:dict, z_score_threshold=-2):
    """
    processes the tags_and_quotes dictionary 
    so that tags appearing in the tag_z_scores dict with z_score less than z_score_threshold
    are combined into a single tag 
    """
    assert "z_score" in tag_z_scores[list(tag_z_scores.keys())[0]].keys(), "entries in tag_z_scores need a z_score field"
    assert type(tags_and_quotes[list(tags_and_quotes.keys())[0]]) == list, "entries in tags_and_quotes should map tags to lists of quotes"
    
    merged_tags = {}
    for tag in tags_and_quotes.keys():
        # look for this tag in the tag_z_scores dict
        for tag1_tag2 in tag_z_scores.keys():
            tags = tag1_tag2.split("_")
            assert len(tags) == 2, f"Tag with unexpected extra underscore {tag1_tag2}"
            # use a threshold
            if (tag_z_scores[tag1_tag2]["z_score"] < z_score_threshold) and ((tag == tags[0]) or (tag == tags[1])):
                print(f"Got tag merge. {tags[0]} and {tags[1]}")# on {tag} for {tag1_tag2} : {tag_z_scores[tag1_tag2]['z_score']}  {tag_z_scores[tag1_tag2]['needs_merge']}") 
                merged_tag = tags[0] + "_" + tags[1]
                if merged_tag not in merged_tags:
                    print(f"Adding {tags[1]}'s {len(tags_and_quotes[tags[1]])} quotes to {merged_tag}")
                    merged_tags[merged_tag] = tags_and_quotes[tags[0]]
                    merged_tags[merged_tag].append(tags_and_quotes[tags[1]])    
            else:
                # no merge needed, just add the individual tag to the 
                # merged tags
                merged_tags[tag] = tags_and_quotes[tag]
                    
    return merged_tags

## Import the documents to a collection
if __name__ == "__main__":
    print("Gettting tags from a collection and summarising")
    print(sys.argv)
    assert len(sys.argv) == 5, "Usage: python script.py  json_tags_to_quotes_File json_tag_z_score_file json_merged_tag_file csv_codebook"
    # print(sys.argv)
    # collection_name = sys.argv[1]
    json_naive_tag_file = sys.argv[1] # read this in 
    json_tag_z_score_file = sys.argv[2] # write this out 
    json_merged_tag_file = sys.argv[3] # write this out
    codebook_csv_file = sys.argv[4] # write this out 
    
    assert os.path.exists(json_naive_tag_file), f"Cannot find tag input file {json_naive_tag_file}"
    
    # extract the 'naive' tags from step 2a
    with open(json_naive_tag_file) as f:
        j_in_str = f.read()
    tags_to_quotes = json.loads(j_in_str)
    tags_to_quotes = merge_tags_on_case(tags_to_quotes)
    tag_list = [t for t in tags_to_quotes.keys()]#[1:5]

    if os.path.exists(json_tag_z_score_file):
        print("Z scores already computed. Skipping calculation")
        with open(json_tag_z_score_file) as f:
            jdata = f.read()
        tag_z_scores = json.loads(jdata)
    else:
        print(f"Got {len(tag_list)} tags. Computing embeds ")
        # build a semantic distance matrix between all tags
        # tags_to_embs = compute_tag_embeddings(tag_list)
        tags_to_embs_and_descs = compute_tag_embeddings_via_description(tag_list)
        # now write out the code book to disk, since we generated it
        # the codebook can be a CSV 
        print("computing tag distances")
        tag_descs = [tags_to_embs_and_descs[t]["description"] for t in tags_to_embs_and_descs]
        print(f"writing codebook to {codebook_csv_file} ")
        codebook_df = pd.DataFrame({"tag":tag_list, "description":tag_descs})
        codebook_df.to_csv(codebook_csv_file)
        # now throw away the descrptions
        tags_to_embs = {}
        for t in tag_list: tags_to_embs[t] = tags_to_embs_and_descs[t]["embedding"]

        tag_dists = compute_tag_distances(tag_list, tags_to_embs)
        tag_z_scores = compute_z_scores(tag_dists)
        # dump z scores so can human sanity check em 
        print(f"Writing z scores to {json_tag_z_score_file}")
        jdata = json.dumps(tag_z_scores)
        with open(json_tag_z_score_file, 'w') as f:
            f.write(jdata)

    # now merge the tags that are significantly similar 
    print("Merging similar tags...")
    # this removes repeates of tag1_tag2 + tag2_tag1 format to simplify things 
    # but it ony gives us a list of tags
    minimal_tag_pair_list = remove_repeated_tags([pair for pair in tag_z_scores.keys()])
    # take that minimal list of tags 
    # and use it to rebuild the tag z scores file??
    tag_z_scores_clean = {}
    for tag_pair in minimal_tag_pair_list:
        tag_z_scores_clean[tag_pair] = tag_z_scores[tag_pair]
    
    # now the clever bit where we do the semantic merge 
    # which will combine the quotes for similar tags under single tag
    refined_tags = semantic_tag_merge(tag_z_scores=tag_z_scores_clean, 
                                      tags_and_quotes=tags_to_quotes, 
                                      z_score_threshold=-4.6)
    # now write the refined tags to disk
    print(f"Writing merged tags to {json_merged_tag_file}")

    print(f"Got merged tags. Tag count: {len(refined_tags.keys())}")
    for t in refined_tags.keys():
        print(f"{t} has {len(refined_tags[t])} quotes")
    
    json_str = json.dumps(refined_tags)
    with open(json_merged_tag_file, 'w') as f:
        f.write(json_str)
    
