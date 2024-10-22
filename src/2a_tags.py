
### This script does a naive tagging process
### where it chops docs into frags and generates tags for each 
### frag with no concern for previously generated tags 

import sys
import ta_utils
import json
import os 





## Import the documents to a collection
if __name__ == "__main__":
    # note that we do not use the word code when working with llms
    # as that tends to trigger generation of computer code
    print("Creating tags for all fragments (aka the codebook)")

    assert len(sys.argv) == 5, f"Usage python script.py collection_id frag_length frag_overlap json_outfile"
    # id for doc collection in the webui system
    collection_id = sys.argv[1]
    # length of fragments 
    frag_len = int(sys.argv[2])
    # frag overlap
    frag_overlap = int(sys.argv[3])
    # json output file
    jfile = sys.argv[4]
    # assert os.path.exists(json) == False, f"json out file seems to already exist so aborting {jfile} " 
    # get the docs from the collection
    assert ta_utils.does_collection_id_exist(collection_id), f"Cannot find a collection with id {collection_id}"
    doc_ids = ta_utils.get_docs_in_collection(collection_id)
    all_tags = {} # this will be {"tag_text":["doc_frag1", "doc_frag2"]} etc.
    # iterate over docs
    for doc_id in doc_ids:
        print(f"Processing {doc_id}")
        # get doc content
        doc_str = ta_utils.get_doc_contents(collection_id, doc_id)
        # split the doc to fragments
        frags = ta_utils.split_text(doc_str, frag_len, frag_overlap)
        # tag the fragments 
        print(f"Frag count for doc: {len(frags)}")
        for frag in frags:
            tags = ta_utils.generate_tags(frag)
            # add tags to all tags, avoiding repeated tags
            print(f"Tag count for frag {len(tags)}")
            for t in tags:
                if t not in all_tags.keys():
                    all_tags[t] = []
                all_tags[t].append(frag)
    # now we have our first phase tags.
    # write to a mega json file (or ideally do something better ... )
    j_data = json.dumps(all_tags)
    print(f"Writing results to {jfile}")
    with open(jfile, 'w') as f:
        f.write(j_data)
    