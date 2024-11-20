import sys
import os 

import ta_utils

## Import the documents to a collection
if __name__ == "__main__":
    print("Creating a collection and importing documents")
    assert len(sys.argv) == 3, "Usage: python script.py document_folder collection_name"
    folder = sys.argv[1]
    assert os.path.exists(folder), f"Imput Folder not found {folder}"
    collection_name = sys.argv[2]
    extension = "docx"
    file_list = ta_utils.get_files_in_folder(folder, extension)
    assert len(file_list) > 0, f"Folder {folder} does not contain any {extension} files"
    print(f"Found {len(file_list)} files. Creating collection first")

    # now create the collection
    # does it already exist?
    collection_id = ta_utils.get_collection_id(collection_name)
    # does not exist- create it 
    if collection_id == None:
        collection_id = ta_utils.create_collection(collection_name)
    
    print(f"Got collection id {collection_id}")
    # assert False # kill it 
    for f in file_list:
        print(f"Adding {f} to coll {collection_id}")
        f_id = ta_utils.add_doc_to_db(f) # create it 
        ta_utils.add_doc_to_collection(collection_id=collection_id, file_id=f_id) # put it in the coll

    # verify the docs are there
    docs = ta_utils.get_docs_in_collection(collection_id)
    assert len(docs) >= len(file_list), f"Found {len(docs)} in collection but wanted at least {len(file_list)}"
    print(f"Added {len(file_list)} docs to collection {collection_id}")