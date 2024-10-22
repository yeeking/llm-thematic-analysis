import sys
import os 

import ta_utils

def test_create():
    coll_name = "test_coll"
    collection_id = ta_utils.get_collection_id(coll_name)
    if collection_id==None:
        collection_id = ta_utils.create_collection(collection_name=coll_name)
    
def test_delete():
    coll_name = "test_coll"
    while collection_id != None:
        collection_id = ta_utils.get_collection_id(coll_name)
        print("deleting ", collection_id)
        ta_utils.delete_collection(collection_id)
    
def test_add_docs():
    coll_name = "test_coll"
    collection_id = ta_utils.get_collection_id(coll_name)
    file_list = ta_utils.get_files_in_folder("../data/docs", "docx")

    for f in file_list:
        print(f"Adding {f} to coll {collection_id}")
        f_id = ta_utils.add_doc_to_db(f)
        ta_utils.add_doc_to_collection(collection_id=collection_id, file_id=f_id)
    
def test_frag():
    text = "the rain in spain falls mainly on the plane"
    frags = ta_utils.split_text(text, 5, 10)
    assert frags[0] == text[0:5]
    assert frags[1] == text[15:20]
    for f in frags:
        assert len(f) == 5, f"frag not 5 steps"
    print(frags)

## Import the documents to a collection
if __name__ == "__main__":
    test_frag()


