
### This script does a naive tagging process
### where it chops docs into frags and generates tags for each 
### frag with no concern for previously generated tags 

import sys
import ta_utils
import json
import os 
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
import re 
import pandas as pd 

def doc_to_frags_semantic(doc_id:str):
    """
    load the sent doc_id from the server, get contents and fragment it
    """
    print(f"Fragging and tagging doc {doc_id}")
    doc_text = ta_utils.get_doc_contents(doc_id)
    print(f"Got text of len {len(doc_text)}")
    # frags = ta_utils.split_text(doc_text, frag_len, frag_hop)
    frags = ta_utils.split_text_semantic(doc_text)
    # tag the fragments 
    print(f"Frag count for doc: {len(frags)}")
    return frags

def doc_to_frags_sentence(doc_id:str, n_sentences=3, n_overlap_sentences=1):
    """
    load the sent doc_id from the server, get contents and fragment it
    """
    print(f"Fragging and tagging doc {doc_id}")
    text = ta_utils.get_doc_contents(doc_id)
    frags = split_text_on_sentences(text, n_sentences,n_overlap_sentences)
    return frags


def split_text_on_sentences(text, n_sentences=3, n_overlap_sentences=1):
    # Split the text into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    # List to hold the fragments
    fragments = []
    
    # Loop to create fragments with the required overlap
    for i in range(0, len(sentences), n_sentences - n_overlap_sentences):
        fragment = sentences[i:i + n_sentences]
        if len(fragment) < n_sentences:  # Stop if fragment is shorter than desired length
            break
        fragments.append(" ".join(fragment))
    
    return fragments


# def split_text_on_sentences(text, n_sentences=3, n_overlap_sentences=1):
#     sentences = sent_tokenize(text)
#     # List to hold the fragments
#     fragments = []
#     # Loop to create fragments with the required overlap
#     for i in range(0, len(sentences), n_sentences - n_overlap_sentences):
#         fragment = sentences[i:i + n_sentences]
#         if len(fragment) < n_sentences:  # Stop if fragment is shorter than desired length
#             break
#         fragments.append(" ".join(fragment))    
#     return fragments

def frag_to_tag(frag:str, all_tags:dict, model):
    """
    process the sent list of frags by extracting tags and storing them 
    into all_tags
    """

    # print(f"***Getting tags for \n\n{frag} \n\n")
    tags = ta_utils.generate_tags(frag, model=model)
    # add tags to all tags, avoiding repeated tags
    for t in tags:
        if t not in all_tags.keys():
            all_tags[t] = []
        all_tags[t].append(frag)
    

# def frags_to_tags(frags:list, all_tags:dict, save_mode:bool, jfile:str, model="llama3.2:3b-instruct-q8_0"):
#     """
#     process the sent list of frags by extracting tags and storing them 
#     into all_tags
#     """
#     f_ind = 0
#     # model = "llama3.1:70b"
#     for frag in frags:
#         print(f"Getting tags for frag {f_ind} of {len(frags)} with model {model}")
#         # print(f"***Getting tags for \n\n{frag} \n\n")
#         tags = ta_utils.generate_tags(frag, model=model)
#         # add tags to all tags, avoiding repeated tags
#         print(f"Tag count for frag {len(tags)}")
#         for t in tags:
#             if t not in all_tags.keys():
#                 all_tags[t] = []
#             all_tags[t].append(frag)
#         f_ind = f_ind + 1
#         if save_mode:
#             j_data = json.dumps(all_tags)
#             with open(jfile, 'w') as f:
#                 f.write(j_data)

def save_tags_to_json(all_tags, jfile):
    j_data = json.dumps(all_tags)
    with open(jfile, 'w') as f:
        f.write(j_data)

## Import the documents to a collection
if __name__ == "__main__":
    print("Gettting tags from a collection and summarising")
    assert len(sys.argv) == 6, "Usage: python script.py collection_name frag_len frag_hop json_outfile model"
    print(sys.argv)
    collection_name = sys.argv[1]
    frag_len = int(sys.argv[2])
    frag_hop = int(sys.argv[3])
    jfile = sys.argv[4]
    # model = "llama3.1:70b"
    # model = "llama3.2:3b-instruct-q8_0"
    # model=  "llama3.1:8b"
    model = sys.argv[5]
    # do a quick test on the file
    with open(jfile, 'w') as f:
        f.write("")
    assert os.path.exists(jfile), f"Cannot write to file {jfile}"
    
    collection_id = ta_utils.get_collection_id(collection_name)
    assert collection_id != None, f"Collection not found: {collection_name}"
    print(f"Proceeding with collection {collection_name} id {collection_id}, writing to file {jfile}")
    docs = ta_utils.get_docs_in_collection(collection_id)
    assert len(docs) > 0, f"Collection does not contain any docs"

    # generate all frags first so can do a progress bar
    all_frags = {}
    total_frags = 0
    for doc_id in docs:
        # all_frags[doc_id] = doc_to_frags_semantic(doc_id)
        all_frags[doc_id] = doc_to_frags_sentence(doc_id, frag_len, frag_hop)
        
        total_frags = total_frags + len(all_frags[doc_id])
    print(f"Total frags: {total_frags}")
    # frags_to_tags(frags.values(), all_tags, save_mode=True, jfile=jfile, model=model)
    
    all_tags = {}
    done_frags = 0
    with tqdm(total=total_frags, desc="Processing Fragments") as pbar:
        for doc_id in docs[0:1]:
            frags = all_frags[doc_id]
            for f in frags[0:1]:
                # print(f)
                frag_to_tag(f, all_tags, model)
                done_frags = done_frags + 1
            # frags_to_tags(frags, all_tags, save_mode=True, jfile=jfile, model=model)
                # print(f"Done {done_frags} of {total_frags}")
                pbar.update(1)  # Update the progress bar by one step
        
            # save once per doc
            # save_tags_to_json(all_tags, jfile)
    print(f"Saving data to {jfile}")
    save_tags_to_json(all_tags, jfile)
    # now save to csv as well
    df = pd.DataFrame({
        "tag":list(all_tags.keys()),
        "quotes":[json.dumps(all_tags[k]) for k in all_tags.keys()]
    })
    df.to_csv(jfile + ".csv")