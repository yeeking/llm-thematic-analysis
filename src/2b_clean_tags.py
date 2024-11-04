### does some basic cleaning and merging of tags
### - make them lower case
### - remove any non-alphanumeric characters from the start

import sys
import os 
import pandas as pd 
import json
import ta_utils


def save_tags_to_json(all_tags:dict, jfile):
    j_data = json.dumps(all_tags)
    with open(jfile, 'w') as f:
        f.write(j_data)

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python script.py tag_json_file"
    print(sys.argv)

    tag_file = sys.argv[1]
    assert os.path.exists(tag_file), f"Cannot find tag file {tag_file}"
    
    with open(tag_file) as f:
        jstr = f.read()
    data = json.loads(jstr)

    clean_data = {}
    for tag in data.keys():
        clean_tag = ta_utils.clean_tag(tag)
        if clean_tag not in clean_data.keys():# allows for merging
            clean_data[clean_tag] = []
        clean_data[clean_tag].extend(data[tag]) # append quotes to existing list which might be empty at first

    print(f"Reduced tags from {len(data.keys())} to {len(clean_data.keys())}")
    out_json = tag_file[0:-5] + "_cleaned.json"
    out_csv = tag_file[0:-5] + "_cleaned.csv"

    df = pd.DataFrame({
        "tag":list(clean_data.keys()),
        "quotes":[json.dumps(clean_data[k]) for k in clean_data.keys()]
    })
    print(f"Saving clean tags to {out_json} and {out_csv}")

    j_data = json.dumps(clean_data)
    with open(out_json, 'w') as f:
        f.write(j_data)

    df.to_csv(out_csv)
    