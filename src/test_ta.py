import sys
import os 

import ta_utils

## Import the documents to a collection
if __name__ == "__main__":
    coll_name = "my_docs"
    collection_id = 0
    while collection_id != None:
        collection_id = ta_utils.get_collection_id(coll_name)
        print("deleting ", collection_id)
        ta_utils.delete_collection(collection_id)
    
    # print(f"coll id for {coll_name} is {collection_id}")
# 