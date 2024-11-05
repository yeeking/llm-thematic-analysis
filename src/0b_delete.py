import sys
import os 

import ta_utils

## Import the documents to a collection
if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python script.py collection_name"
    collection_name = sys.argv[1]
    print(f"deleting collections with name {collection_name}")
    ta_utils.delete_collections_with_name(collection_name)

    