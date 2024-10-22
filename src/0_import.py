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
    collection_id = ta_utils.create_collection(collection_name)
    
    assert collection_id != "", f"Collection id looks bad. Got: '{collection_id}'"

    print(f"Got collection id {collection_id}")
    for f in file_list:
        print(f"Inserting {f} to {collection_id}")
        assert os.path.exists(f), f"Error: cannot find file {f}"
        ta_utils.add_doc_to_collection(collection_id, f)