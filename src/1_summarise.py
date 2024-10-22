### generate summaries of the documents 
### (familiarising yourself with your data)

import sys
import os 

import ta_utils

if __name__ == "__main__":
    print("Gettting docs from a collection and summarising")
    assert len(sys.argv) == 2, "Usage: python script.py collection_name"
    collection_name = sys.argv[1]
    collection_id = ta_utils.get_collection_id(collection_name)
    assert collection_id != None, f"Collection not found: {collection_name}"
    print(f"Proceeding with collection {collection_name} id {collection_id}")
    docs = ta_utils.get_docs_in_collection(collection_id)
    assert len(docs) > 0, f"Collection does not contain any docs"
    for doc_id in docs:
        print(f"Summarising doc {doc_id}")
        doc_text = ta_utils.get_doc_contents(doc_id)
        print(f"Got text of len {len(doc_text)}")
        summary = ta_utils.get_text_summary(doc_text)
        print(f"Got summary\n\n{summary}")