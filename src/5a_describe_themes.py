# 5a_describe_themes.py
# this script prompts the LLM
# to describe each theme in terms of its tags and quotes 
# 


import sys
import os 
import pandas as pd
import numpy as np 
import json 
import ta_utils
from tqdm import tqdm

def get_prompt(theme_title, tags, quotes):
    tags_as_csv = ",".join(tags) # 
    quotes_as_csv = ",".join(quotes)
    prompt = f"""
    Here is some information about a theme:
    Theme title: {theme_title}
    Tags describing the theme: {tags_as_csv}
    Quotes attached to the theme: 
   
    {quotes_as_csv}

    Please write a description of the theme, taking into consideration the tags and quotes provided. When possible, identify elements that support and would help answering this research question: 

    "Which ethical and practical perceptions do academics report concerning the introduction of LLM-powered tools to assessment workflows and do they feel differ ently about tools that carry out tasks academics cannot possibly do and tools that carry out tasks academics currently do?"

    Can you also directly include quotes in case you identify ethical and practical perceptions?
    """
    return prompt

if __name__ == "__main__":
    assert len(sys.argv) == 2, "usage: python script.py themes_and_quotes.csv"
    theme_data_csv = sys.argv[1]
    assert os.path.exists(theme_data_csv), f"Cannot find themes and quotes csv {theme_data_csv}"
    context_len = 8192 # max tokens for gemma

    data = pd.read_csv(theme_data_csv)
    # verify necessary keys are in the CSV 
    for f in ["theme", "quotes", "tag"]:
        assert f in data.keys(), f"No {f} field in {data.keys()}"
    
    themes = data["theme"].unique()
    all_prompts = {}
    for theme in themes:
        matches = data[data["theme"] == theme]
        tags = matches["tag"].unique()
        quotes = matches["quotes"].unique() # gets a list of json strings with quote lists in 
        total_q = 0
        uniq_q = set() # unqiue set of quotes for the theme can go here 
        for quote_json in quotes:
            quote_list = json.loads(quote_json)
            for q in quote_list:
                uniq_q.add(q)
                total_q += 1

        
        prompt = get_prompt(theme, tags, uniq_q)
        prompt_tokens = len(prompt.split(" "))
        print(f"{len(matches)} rows for {theme} with {len(tags)} tags and {len(uniq_q)} of total {total_q} quotes. Prompt length {prompt_tokens}")

        if prompt_tokens > context_len:
            q_list = list(uniq_q)
            quotes1 = q_list[0:int(len(q_list)/2)]
            quotes2 = q_list[int(len(q_list)/2):]
            print(f"Split qoutes to {len(quotes1)} and {len(quotes2)}")
            prompt1 = get_prompt(theme, tags, quotes1)
            prompt2 = get_prompt(theme, tags, quotes2)
            assert len(prompt1.split(" ")) < context_len, f"Oh dear mega theme {theme}"
            assert len(prompt2.split(" ")) < context_len, f"Oh dear mega theme {theme}"
            
            
            all_prompts[theme] = []
            all_prompts[theme].append(prompt1)
            all_prompts[theme].append(prompt2)
        else:
            all_prompts[theme] = []
            all_prompts[theme].append(prompt)
            
    # now lets run the prompts
    theme_descriptions = []
    with tqdm(total=len(themes), desc="Processing Tags") as pbar:
        for theme in all_prompts.keys():
            prompts = all_prompts[theme]
            res = ""
            for p in prompts:
                res = res + "\n" + ta_utils.get_chat_completion_lmstudio(p)
            theme_descriptions.append(res) # same order as themes
            pbar.update(1)
            # break
    # write to CSV
    dataf = pd.DataFrame({"theme":themes, "theme_description":theme_descriptions})
    dataf.to_csv("theme_descriptions.csv")

    
    
