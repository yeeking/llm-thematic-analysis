import requests 
import os 
import json
from nltk.tokenize import TextTilingTokenizer
import ast 
import ollama
import numpy as np
import re 
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn import preprocessing
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.metrics import calinski_harabasz_score
import pandas as pd 
from openai import OpenAI

## utiltiy functions for thematic analysis

# based on live api docs: 

# Main	http://127.0.0.1:8080/docs
# WebUI	http://127.0.0.1:8080/api/v1/docs
# Ollama	http://127.0.0.1:8080/ollama/docs
# OpenAI	http://127.0.0.1:8080/openai/docs
# Images	http://127.0.0.1:8080/images/api/v1/docs
# Audio	http://127.0.0.1:8080/audio/api/v1/docs
# RAG	http://127.0.0.1:8080/retrieval/api/v1/docs

## in open webui this is not super secret
## so i'm putting it here. If we talk to other apis
## this should go in the bash envt. 
# API_TOKEN = "sk-43130b6612624d6aaaecb5fa980fda0c" # tp42
# API_TOKEN = "sk-43130b6612624d6aaaecb5fa980fda0c" # tp42
API_TOKEN = "sk-329c26835f524e168d34eb5cc4ac5dad" # mac-studio
# API_TOKEN = "sk-1b2e731745ce43b99d2f1cf4a0edd895" # wispa
BASE_URL = "http://127.0.0.1:8080/" # Replace with your Open WebUI instance URL


def get_api_headers():
    global API_TOKEN
    headers = {
        'Authorization': f'Bearer {API_TOKEN}',
        'Content-Type': 'application/json'
    }
    return headers 

def get_files_in_folder(folder: str, extension: str):
    """
    Return a list of file paths in the sent folder that have the specified extension
    """
    return [os.path.join(folder, file) for file in os.listdir(folder) 
            if os.path.isfile(os.path.join(folder, file)) and file.endswith(extension)]


def get_chat_completion(prompt:str, model:str, max_tokens=100):
    """
    carry out a chat completion using the webui layer on top of ollama
    """
    global BASE_URL
    url = f'{BASE_URL}/api/chat/completions'
    headers = get_api_headers()
    data = {
        "model":model, 
        "keep_alive":240, # don't unload the model for 240 s in case another request..
        "num_ctx":8192,  # this makes llama 3.1 60b work in ollama - hope it gets sent through
    #   "model": "llama3.2:latest",
    # "model":"llama3.1:70b", 
    #   "model":"llama3.1:latest", 
      "messages": [
        {
          "role": "user",
          "content": f"{prompt}"
        }
      ]
    }
    response = requests.post(url, headers=headers, json=data)
    data = response.json()
    assert "detail" not in data.keys(), f"Response bad: {data['detail']}"

    text = data["choices"][0]["message"]["content"]
    return text


def get_chat_completion_lmstudio(prompt:str):
    """
    get_chat_completion but talking to the 'openai-like' API of lm studio instead
    of webui
    """
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    completion = client.chat.completions.create(
        model="lmstudio-community/Llama-3.1-Nemotron-70B-Instruct-HF-GGUF", # it ignores the model anyways
        # model="lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF",
        # model=model, 
        messages=[
        # {"role": "system", "content": "Always answer in rhymes."},
        {"role": "user", "content": prompt}, 
        ],
        max_tokens=200 # stop it from ad nauseam

        # temperature=0.7,
    )
    # print(f"Got lmstudio result {completion}")
    result = completion.choices[0].message.content
    return result 

def list_collections():
    """
    returns a list of doc collections currently on the server
    """
    doc_collections = []
    return doc_collections

def get_collection_list():
    """
    returns a list of available collections
    """
    global BASE_URL
    url = f'{BASE_URL}/api/v1/knowledge'
    headers = get_api_headers()
    response = requests.get(url, headers=headers)
    k_list = response.json()
    assert type(k_list) is list, f"Expected a list of collections but got {k_list}"
    # assert "detail" not in k_list.keys(), f"Looks like the request failed : {k_list}"
    return k_list

def get_collection_id(name:str):
    """
    check if a collection exists.
    http://127.0.0.1:8080/api/v1/docs#/
    @return True or False
    """
    global BASE_URL
    url = f'{BASE_URL}/api/v1/knowledge'
    headers = get_api_headers()
    response = requests.get(url, headers=headers)
    k_list = response.json()
    assert type(k_list) is list, f"Expected a list of collections but got {k_list}"
    # assert "detail" not in k_list.keys(), f"Looks like the request failed : {k_list}"
    collection_id = None
    for k in k_list:
        if k["name"] == name:
            collection_id = k["id"]
            break
    return collection_id
    
def delete_collection(collection_id:str):
    """
    delete the collection and the docs for the sent id 
    /knowledge/{id}/delete
    """
    global BASE_URL
    url = f'{BASE_URL}/api/v1/knowledge/{collection_id}/delete'
    headers = get_api_headers()
    response = requests.delete(url, headers=headers)
    response = response.json()
    print(response)

def delete_collections_with_name(collection_name:str):
    """
    CAREFUL... deletes all collections with the sent name 
    """
    collection_id = 0
    while collection_id != None:
        collection_id = get_collection_id(collection_name)
        print("deleting ", collection_id)
        delete_collection(collection_id)

def create_collection(collection_name:str):
    """
    create a collection and return its id
    """
    global BASE_URL
    url = f'{BASE_URL}/api/v1/knowledge/create'
    headers = get_api_headers()
    collection_id = ""
    data = {
        "name": collection_name,
        "description": "string",
        "data": {}
    }
    response = requests.post(url, headers=headers, json=data)
    response = response.json()
    assert "detail" not in response.keys(), f"Response bad: {response['detail']} key: '{API_TOKEN}'"
    return response["id"] 
    
def add_doc_to_db(file_path):
    """
    create a file in the database from a local file
    you need to do this before you can add it to a collection
    returns a file_id
    http://127.0.0.1:8080/api/v1/docs#/files/upload_file_files__post
    """
    global API_TOKEN, BASE_URL
    url = f'{BASE_URL}/api/v1/files/'
    # different headers for this one as we are posting binary data
    headers = {
        'Authorization': f'Bearer {API_TOKEN}',
        'Accept': 'application/json'
    }
    print(headers)
    files = {'file': open(file_path, 'rb')}
    response = requests.post(url, headers=headers, files=files)
    response = response.json()
    assert "detail" not in response.keys(), f"File add failed for some reason {response}"
    file_id = response["id"]
    return file_id

def add_doc_to_collection(file_id, collection_id):
    """
    add a doc to the sent collection and return its id
    http://127.0.0.1:8080/api/v1/docs#/knowledge/add_file_to_knowledge_by_id_knowledge__id__file_add_post
    Note that is automatically adds meta data to the file like this:
        "meta": {
        "name": "omar_exam_setter.docx",
        "content_type": null,
        "size": 44914,
        "path": "... /python3.11/site-packages/open_webui/data/uploads/0ee6481c-afa8-488b-817c-f46cf585d7bf_omar_exam_setter.docx",
        "collection_name": "aa3e43dc-0f10-4f24-9752-16614a3d549a"
      },
    """
    global BASE_URL
    url = f'{BASE_URL}/api/v1/knowledge/{collection_id}/file/add'
    headers = get_api_headers()
    data = {'file_id': file_id}
    response = requests.post(url, headers=headers, json=data)
    return response.json()
    doc_id = ""
    return doc_id

def get_docs_in_collection(collection_id:str):
    """
    get a list of doc ids from the sent collection id
    """
    global BASE_URL
    url = f'{BASE_URL}/api/v1/knowledge/{collection_id}'
    headers = get_api_headers()
    response = requests.get(url, headers=headers)
    response = response.json()
    assert "detail" not in response.keys(), f"Response bad: {response['detail']}"
    assert ("data" in response.keys()) and ("file_ids" in response["data"].keys()), f"Collection data has no files. Here's what I got back: {response}"
    return response["data"]["file_ids"]
    
def get_doc_metadata(doc_id):
    """
    return the metadata:
    "meta": {
        "name": "omar_exam_setter.docx",
        "content_type": null,
        "size": 44914,
        "path": "... /python3.11/site-packages/open_webui/data/uploads/0ee6481c-afa8-488b-817c-f46cf585d7bf_omar_exam_setter.docx",
        "collection_name": "aa3e43dc-0f10-4f24-9752-16614a3d549a"
      },
    """
    global API_TOKEN, BASE_URL
    url = f'{BASE_URL}/api/v1/files/{doc_id}'
    headers = get_api_headers()
    response = requests.get(url, headers=headers)
    response = response.json()
    assert "meta" in response.keys(), f"No meta in doc data {doc_id}: {response['detail']} on url {url}"
    return response["meta"]


def get_doc_contents(doc_id):
    """
    get the doc contents as a string
    """
    global API_TOKEN, BASE_URL
    url = f'{BASE_URL}/api/v1/files/{doc_id}/data/content'
    headers = get_api_headers()
    response = requests.get(url, headers=headers)
    response = response.json()
    assert "content" in response.keys(), f"No content in doc {doc_id} on url {url}"
    return response["content"]

def get_data_filename(model, collection):
    """
    converts a model name and collection name into an alphanumeric filename
    e.g.  'llama3.2:latest' 'ai-edu-all'
    goes to aieduallllama32latest
    """
    dfile = collection + "-" + model
    dfile = re.sub(r'[^a-zA-Z0-9]', '', dfile)
    return dfile


def get_text_summary(text:str):
    """
    return a summary of the sent doc
    """
    prompt = f"Please summarise the following text in a single paragraph with bullet points {text}"
    summary = get_chat_completion(text)
    return summary

def split_text_on_sentences(text, n_sentences=3, n_overlap_sentences=1):
    # Split the text into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)
    # List to hold the fragments
    fragments = [] 
    # Loop to create fragments with the required overlap
    for i in range(0, len(sentences), n_sentences - n_overlap_sentences):
        fragment = sentences[i:i + n_sentences]
        if len(fragment) < n_sentences:  # Stop if fragment is shorter than desired length
            break
        fragments.append(" ".join(fragment))
    
    return fragments

def doc_to_frags_sentence(doc_id:str, n_sentences=3, n_overlap_sentences=1):
    """
    load the sent doc_id from the server, get contents and fragment it
    """
    print(f"Fragging and tagging doc {doc_id}")
    text = get_doc_contents(doc_id)
    frags = split_text_on_sentences(text, n_sentences,n_overlap_sentences)
    return frags

def doc_to_frags_semantic(doc_id:str):
    """
    load the sent doc_id from the server, get contents and fragment it
    """
    doc_text = get_doc_contents(doc_id)
    # frags = ta_utils.split_text(doc_text, frag_len, frag_hop)
    frags = split_text_semantic(doc_text)
    # tag the fragments 
    return frags

def split_text(text:str, frag_length:int, hop_size:int):
    """
    split the sent text into chunks of the sent length with the sent overlap 
    might want to use the webui api to do this or to pull frags out of the collection
    """
    frags = []
    start = 0
    end = start + frag_length
    while end < len(text):
        frags.append(text[start:end])
        start = end + hop_size
        end = start + frag_length
        
    return frags

def split_text_semantic(text:str):
    """
    Split using the NLTK TextTilingTokenizer - from its docs: 
    Tokenize a document into topical sections using the TextTiling algorithm. This algorithm detects subtopic shifts based on the analysis of lexical co-occurrence patterns.

    The process starts by tokenizing the text into pseudosentences of a fixed size w. Then, depending on the method used, similarity scores are assigned at sentence gaps. The algorithm proceeds by detecting the peak differences between these scores and marking them as boundaries. The boundaries are normalized to the closest paragraph break and the segmented text is returned.
    """
    tt_tokenizer = TextTilingTokenizer()
    segments = tt_tokenizer.tokenize(text)
    return segments



def generate_tags(text:str, model:str, lm_studio_mode=False, bad_tags_file='bad_tags.txt', tag_clean_model="llama3.2:3b-instruct-q8_0"):
    """
    generate a list of tags for the sent text
    """
    # prompt = f"The following text is a an extract from an interview. Here is the text: \"{text}\". I would like you to generate one, two, three or four tags which describe the text. The tags can have one, two or three words and should describe the text and also identify the intention, sentiment or emotional content of the text. An example of such a tag is: \"happy about the weather\".  You do not need to explain the tags, just print out the list of tags."
    prompt = f"The following text is an extract from an interview: \"{text}\". (That is the end of the extract). I would like you to generate one or more useful tags to describe the text. The tag should contain a sense of the drive, sentiment or emotion of the text and it should clearly identify the topic, subject or object of the text. For example \"matter-of-fact response\" is not a useful tag because it does not contain the topic. \"doubtful of accuracy\" is not a useful tag because it is not clear what the accuracy refers to. I do not want tags that describe the flow of the conversation, for example \"casual conversation closure\" is not useful because it describes the conversation not the content of the conversation. If the text seems to contain the interviewer Andrea asking a question, then you can ignore that text and just concentrate on the answer to the question if the answer is there. Here are examples of useful tags: \"acceptance of technological supplementation\" is an example of a helpful and useful tag because it identifies the direction and object. \"concern for unintentional offenders\" is also a useful tag because it identifies the drift and the target. Please write the most useful tags you can as a bullet point list. You do not need to explain why you chose the tags, just print the tags themselves. "
    # print(f"Sending initial prompt {prompt}")
    if lm_studio_mode:
        tags = get_chat_completion_lmstudio(prompt)
    else:
        tags = get_chat_completion(prompt, model)

    # now ask it to format it as json
    prompt = f"Please format the following list of tags into a JSON list format. Only print the tags in the JSON list, do not explain it, do not make it a dictionary. Here is an example of the format: ['tag 1', 'tag 2'] Here are the tags: \"{tags}\""
    tags_raw = get_chat_completion(prompt, model=tag_clean_model)

    # print(f"\n\n***Raw tag data: {tags_raw}")
    # now try for a rough parsing of the data into JSON
    try:
        tags = ast.literal_eval(tags_raw)
        # tags = json.loads(tags_raw)
        if type(tags) is not list:
            tags = []
        
    except: # fail condition is a bit harsh but...
        print(f"Could not parse these tags: \"{tags_raw}\"")
        tags = []
    if tags == []:
        with open(bad_tags_file, 'a') as f:
            f.write("\n\n"+tags_raw+"\n\n")
    return tags


def text_to_embeddings(text):
    """
    generate an embedding for the sent text 
    """
    response = ollama.embeddings(model="mxbai-embed-large", prompt=text, keep_alive=240) # keep model inn memory for 240 s
    embedding = response["embedding"]
    return embedding



# def compute_tag_embeddings_via_description_lmstudio(tag, model):
#     prompt = f"I am working on a qualitative analysis of some interviews wherein I am assigning tags to fragments of text. I have assigned the following tag: '{tag}' to an extract from the interview. Please can you write a short description of what that tag is likely to be about. Only provide the tag description please, not your justification of that description, and do not refer to the tag, just provide the description. "
#     description = get_chat_completion_lmstudio(prompt, model)
#     emb = text_to_embeddings(description)
#     return description, emb


def compute_tag_embeddings_via_description(tag:str, model):
    prompt = f"I am working on a qualitative analysis of some interviews wherein I am assigning tags to fragments of text. I have assigned the following tag: '{tag}' to an extract from the interview. Please can you write a short description of what that tag is likely to be about. Only provide the tag description please, not your justification of that description, and do not refer to the tag, just provide the description. "
    # print(f"Running prompt '{prompt}'")
    # description = get_chat_completion(prompt, model)
    result = ollama.generate(model=model, prompt=prompt, keep_alive=240) # keep alive is measured in seconds and tells ollama to keep the model loaded
    assert "response" in result.keys(), f"ollama response looks bad {result}"
    description = result["response"]
    # print(result)
    # description = "test"
    # print(f"done. Got result '{description}'")
    emb = text_to_embeddings(description)
    return description, emb
    
def clean_tag(tag:str):
    """
    trims any non-alphanumeric characters from the start of the tag, e.g.
    '- the tag' -> 'the tag' and converts to lower case 
    """
    tag = tag.lower()
    tag = re.sub(r'^[^a-zA-Z]+', '', tag)
    return tag 

def clean_tag(tag:str):
    """
    trims any non-alphanumeric characters from the start of the tag, e.g.
    '- the tag' -> 'the tag' and converts to lower case 
    """
    tag = tag.lower()
    tag = re.sub(r'^[^a-zA-Z]+', '', tag)
    return tag 

def get_z_scores(values):
    """
    computes the z-scores for the sent values
    which are the values - mean_value / std_dev_value
    """
    values = np.array(values)
    # Step 1: Calculate the mean and standard deviation of cosine distances
    mean_value = np.mean(values)
    std_dev_value = np.std(values)
    # print(values)
    # print(mean_value, std_dev_value)

    # Step 2: Calculate z-scores for each cosine distance
    z_scores = (values - mean_value) / std_dev_value
    return z_scores 

def get_tag_to_embeddings(csv_filename, embedding_field="embeddings"):
    """
    returns a dictionary mapping tags to mebeddings
    """
    data = pd.read_csv(csv_filename)
    tag_embeddings = {}

    for ind,row in data.iterrows():
        if pd.isna(row["tag"]):
            print(f"Found bad tag. Skipping {row['tag']}")
            continue
        # assert pd.isna(row["tag"]) != True, f"Found a nan tag. {row['tag']} "
        tag_embeddings[row['tag']] = np.array(json.loads(row[embedding_field]))
    return tag_embeddings
    

def get_pca_variances(embeddings):
    """
    returns a dictionary mapping number of PCA components
    to explained variance, e.g. x[100] = 0.95 # 100 components explains 95% of variance
    """
    maxn = len(embeddings)
    pca_components = np.arange(2, maxn, maxn/100) # 10 values from 2 to max components
    print(f"Runniung pca with values {pca_components} on embeeddings shape {embeddings[0]}")
    # convert to normalised version which is how we'll run it later
    embeddings = preprocessing.normalize(embeddings)

    # Dictionary to store cumulative explained variance for each component count
    explained_variances = {}
    target = 0.95
    # Loop over each number of components
    for n_components in pca_components:
        print(f"Doing PCA with {n_components}")
        # Apply PCA with n_components
        n_components = int(n_components)
        pca = PCA(n_components=n_components)
        # print(embeddings)
        pca.fit(embeddings)  # 'embeddings' should be your dataset of vectors

        # Calculate cumulative explained variance
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)[-1]  # Total explained variance for n_components
        explained_variances[n_components] = cumulative_variance

        if cumulative_variance >= target:
            best_n = n_components
            break
        print(f"PCA components: {n_components}, Cumulative explained variance: {cumulative_variance:.4f}")
    
    return explained_variances, n_components



def get_cluster_scores(pca_components, embeddings, cluster_range=[2,4,6,8,10,20]):
    """
    returns a dictionary mapping number of clusters -> davies_bouldin and silhouette scores
    """
    # Apply PCA to reduce dimensions
    if pca_components > 0:
        normed_embeddings = preprocessing.normalize(embeddings)
        pca = PCA(n_components=pca_components)
        # reduce dimensionality
        normed_embeddings = pca.fit_transform(normed_embeddings)
        # convert for cosine distance
    else: 
        normed_embeddings = preprocessing.normalize(embeddings)

    # Dictionary to store Davies-Bouldin and Silhouette scores for each k
    results = {}

    # Loop over each number of clusters (k)
    for k in cluster_range:
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=k, random_state=0)
        cluster_labels = kmeans.fit_predict(normed_embeddings)

        # Calculate Davies-Bouldin Index
        db_index = davies_bouldin_score(normed_embeddings, cluster_labels)
        
        # Calculate Silhouette Score (only if k > 1)
        silhouette_avg = silhouette_score(normed_embeddings, cluster_labels)

        ch_score = calinski_harabasz_score(normed_embeddings, cluster_labels)
        
        
        results[k] = {'davies_bouldin': db_index, 
                      'silhouette': silhouette_avg, 
                      "Calinski_Harabasz":ch_score}
        
        print(f"PCA Dim: {pca_components}, Clusters: {k}, CH score:{ch_score} DB Index: {db_index:.3f}, Silhouette Score: {silhouette_avg:.3f}" if k > 1 else f"PCA Dim: {dim}, Clusters: {k}, DB Index: {db_index:.3f}")

    return results

def cluster_items(embeddings, pca_n, k):
    """
    does a kmeans on the sent embeddings, after preprocessing.normalizing them
    and pca ing them
    returns the labels for the clusters
    """
    # Apply PCA to reduce dimensions
    normed_embeddings = preprocessing.normalize(embeddings)
    pca = PCA(n_components=pca_n)
    # reduce dimensionality
    normed_embeddings = pca.fit_transform(normed_embeddings)

    kmeans = KMeans(n_clusters=k, random_state=0)
    cluster_labels = kmeans.fit_predict(normed_embeddings)
    return cluster_labels

def tags_to_theme(tags, lm_studio_mode, model, title_clean_model='llama3.2:3b-instruct-q8_0'):
    research_questions = """
RQ1: To what extent do academic staff fear that LLM systems will replace or significantly alter their role in exam setting and marking, potentially threatening their job security, autonomy, or ability to exercise professional judgment, and how does this impact their acceptance of these systems? 

RQ2: To what extent do academic staff trust the accuracy and fairness of LLM systems in setting and marking exams, and what are their views on the potential risks and biases associated with these systems, including issues related to biases, transparency, accountability, fairness, false positives/negatives? 
"""
    prompt = f"""I am carrying out a thematic analysis of a set of interviews with academics. In the background, I am interested in investigating different elements of the following research questions:

    {research_questions} 

    I have analysed the interviews and come up with the following list of tags that describe a specific theme I found in the contents of the interviews. There are lots of other themes, but this is the one I am interested in now: 

    {tags} 

    I want you to create me a title for the theme. The title should make connections between the tags and the most relevant aspects of the research questions. The tags will not relate to every aspect of the research questions so the theme title should focus more on the tags. 
    
    The theme name should be concise and specific, conveying what the tags in this theme represent. Ensure the title reflects the specific aspect of the tags captured by this theme, rather than attempting to cover too much or becoming overly complex. Describe the overarching 'essence' of the theme so that it could be summarised in one to two sentences if needed. The title should reflect both the distinct focus of this set of tags and its relation to the research question. You only need to state the theme title – you do not need to explain why you chose that title.  
"""
#     # V2: without research questions
#     prompt = f"""I am carrying out a thematic analysis of a set of interviews with academics. I have analysed the interviews and come up with the following list of tags that describe a specific theme I found in the contents of the interviews. There are lots of other themes, but this is the one I am interested in now: 

#     {tags} 

#     I want you to create me a title for the theme. The theme name should be concise and specific, conveying what the tags in this theme represent. Ensure the title reflects the specific aspect of the tags captured by this theme, rather than attempting to cover too much or becoming overly complex. Describe the overarching 'essence' of the theme so that it could be summarised in one to two sentences if needed. You only need to state the theme title – you do not need to explain why you chose that title.  
# """
    # print(f"{prompt}")
    # result = ta_utils.get_chat_completion(prompt, model)
    if lm_studio_mode:
        result = get_chat_completion_lmstudio(prompt) # lm ignores the model actually
    else:
        result = get_chat_completion(prompt, model)

    # stage to ensure we just get the title 
    prompt = f"Please read the following text and extract the theme title. Present the title in JSON format. Do not explain it, do not make it a dictionary. Here is an example of the format: [\"title\"]. Here is the text: \"{result}\""
    
    title_raw = get_chat_completion(prompt, model=title_clean_model)
    # print(f"Got raw title {title_raw}")
    bad_themes_file = "bad_themes.txt"
    try:
        title_raw = json.loads(title_raw)[0]
    except:
        with open(bad_themes_file, 'a') as f:
            f.write(title_raw + "\n")
        print(f"Problem with theme title. First run generated: '{result}' second run generated {title_raw} ")
    # assert title_raw in result, f"IT made up a title or something else terrible happened. Here's the result\n\n {result} \n\n\n here's what got extracted {title_raw}"
    
    # print(f"Raw title {title_raw}")

    return title_raw 

def compute_z_scores(data):
    """
    Computes z-scores for a list of numbers.

    Parameters:
    data (list or array-like): A list of numerical values.

    Returns:
    list: A list of z-scores corresponding to the input data.
    """
    mean = sum(data) / len(data)
    std_dev = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
    z_scores = [(x - mean) / std_dev for x in data]
    return z_scores


def get_theme_to_ind_lookup(themes):
    # Sort the list of themes
    sorted_themes = sorted(themes)
    # Create the dictionary mapping each theme to its index
    theme_to_index = {theme: index for index, theme in enumerate(sorted_themes)}
    return theme_to_index