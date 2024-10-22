
### This script does a naive tagging process
### where it chops docs into frags and generates tags for each 
### frag with no concern for previously generated tags 

import sys
import ta_utils
import json
import os 


## Import the documents to a collection
if __name__ == "__main__":
    print("Gettting tags from a collection and summarising")
    assert len(sys.argv) == 5, "Usage: python script.py collection_name frag_len frag_hop json_outfile"
    print(sys.argv)
    collection_name = sys.argv[1]
    frag_len = int(sys.argv[2])
    frag_hop = int(sys.argv[3])
    jfile = sys.argv[4]
    # do a quick test on the file
    with open(jfile, 'w') as f:
        f.write("")
    assert os.path.exists(jfile), f"Cannot write to file {jfile}"
    
    collection_id = ta_utils.get_collection_id(collection_name)
    assert collection_id != None, f"Collection not found: {collection_name}"
    print(f"Proceeding with collection {collection_name} id {collection_id}, writing to file {jfile}")
    docs = ta_utils.get_docs_in_collection(collection_id)
    assert len(docs) > 0, f"Collection does not contain any docs"
    for doc_id in docs:
        print(f"Fragging and tagging doc {doc_id}")
        doc_text = ta_utils.get_doc_contents(doc_id)
        print(f"Got text of len {len(doc_text)}")
        all_tags = {}
        frags = ta_utils.split_text(doc_text, frag_len, frag_hop)
        # tag the fragments 
        print(f"Frag count for doc: {len(frags)}")
        for frag in frags:
            print(f"***Getting tags for \n\n{frag} \n\n")
            tags = ta_utils.generate_tags(frag)
            # add tags to all tags, avoiding repeated tags
            print(f"***Got tags\n\n{tags}")
            print(f"Tag count for frag {len(tags)}")
            for t in tags:
                if t not in all_tags.keys():
                    all_tags[t] = []
                all_tags[t].append(frag)
            break
        break
    # now we have our first phase tags.
    # write to a mega json file (or ideally do something better ... )
    j_data = json.dumps(all_tags)
    print(f"Writing results to {jfile}")
    with open(jfile, 'w') as f:
        f.write(j_data)
    