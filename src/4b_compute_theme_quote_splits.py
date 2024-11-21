### for each theme, compute a number between 0 and 1 that 
### says what proportion of its quotes are from either of the two exam/ collusion datasets

import os
import sys
import pandas as pd 
import json
import ta_utils

def get_quotes_for_theme(theme, df):
    """
    combines all quotes for a given theme into a single list of quotes by extracting and converting json from the dataframe
    """
    raw_quotes = df[df["theme"] == theme]["quotes"] # this should be a list of json fragments
    all_quotes = []
    for q in raw_quotes:
        quote_list = json.loads(q)
        all_quotes.extend(quote_list)
    # print(f"found {len(all_quotes)} quotes in {theme}")
    return all_quotes

def get_quote_split(quotes):
    """
    works out how many of the quotes are from '_collusion' data files
    and how many are from '_exam' data files and then computes the ratio 
    format for file names is 'name_dataset.csv_ "
    e.g. 
    myk_collusion.docx
    myk_exam_setter.docx
    """
    collusion = 0
    exam = 0
    for q in quotes:
        parts = q.split("_")
        assert len(parts) > 1, f"Quote looks bad {q}"
        assert (parts[1].startswith("collusion")) or (parts[1].startswith("exam")), f"Did not find dataset marker in {parts[1]}"
        if parts[1].startswith("collusion"):collusion += 1
        elif parts[1].startswith("exam"): exam += 1
    return {"total":(collusion+exam), "collusion":collusion, "exam":exam, "collusion_ratio":collusion/(collusion+exam), "exam_ratio":exam/(collusion + exam)}


if __name__ == "__main__":

    assert len(sys.argv) == 3, f"Usage python script.py tags_themes_and_quotes.csv output_theme_stats_here.csv"

    csv_file = sys.argv[1]
    csv_outfile = sys.argv[2]
    assert os.path.exists(csv_file), f"Cannot find input CSV {csv_file}"
    data = pd.read_csv(csv_file)
    fields = ["theme", "quotes"]
    for f in fields: assert f in data.keys(), f"Data file does not have {f} field"

    # now combine quotes
    themes = sorted(data["theme"].unique())
    data_dict = {"theme_id":[],
                 "theme":[]}
    
    theme_to_id_lookup = ta_utils.get_theme_to_ind_lookup(themes)

    for ind,t in enumerate(themes):
        quotes = get_quotes_for_theme(t, data)
        counts = get_quote_split(quotes)
        # print(f"{t} split: {counts}")
        data_dict["theme_id"].append(theme_to_id_lookup[t])
        data_dict["theme"].append(t)

        for k in counts.keys():
            if k not in data_dict: data_dict[k] = []
            data_dict[k].append(counts[k])

    out_df = pd.DataFrame(data_dict)
    print(f"4b_etc saving csv to {csv_outfile}")
    out_df.to_csv(csv_outfile)