import sys
import os 

import ta_utils

## Import the documents to a collection
if __name__ == "__main__":
    assert len(sys.argv) == 1, "Usage: python script.py"

    collections = ta_utils.get_collection_list()
    for k in collections:
        print(f"Collection {k['name']} has id {k['id']}")
    