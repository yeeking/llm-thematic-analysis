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
API_TOKEN = "sk-43130b6612624d6aaaecb5fa980fda0c" # tp42
# API_TOKEN = "sk-43130b6612624d6aaaecb5fa980fda0c" # tp42
# API_TOKEN = "sk-329c26835f524e168d34eb5cc4ac5dad" # mac-studio
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


def get_chat_completion_lmstudio(prompt:str, model:str):
    """
    get_chat_completion but talking to the 'openai-like' API of lm studio instead
    of webui
    """
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    completion = client.chat.completions.create(
        # model="lmstudio-community/Llama-3.1-Nemotron-70B-Instruct-HF-GGUF",
        # model="lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF",
        model=model, 
        messages=[
        # {"role": "system", "content": "Always answer in rhymes."},
        {"role": "user", "content": prompt}
        ]
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
    
def get_doc_contents(doc_id):
    """
    get the doc contents as a string
    """
    global API_TOKEN, BASE_URL
    url = f'{BASE_URL}/api/v1/files/{doc_id}/data/content'
    headers = get_api_headers()
    response = requests.get(url, headers=headers)
    response = response.json()
    assert "content" in response.keys(), f"No content in doc {doc_id}"
    return response["content"]

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



def generate_tags(text:str, model:str, lm_studio_mode=False, bad_tags_file='bad_tags.txt'):
    """
    generate a list of tags for the sent text
    """
    prompt = f"The following text is a an extract from an interview. Here is the text: \"{text}\". I would like you to generate one, two, three or four tags which describe the text. The tags can have one, two or three words and should describe the text and also identify the intention, sentiment or emotional content of the text. An example of such a tag is: \"happy about the weather\".  You do not need to explain the tags, just print out the list of tags."
    # print(f"Sending initial prompt {prompt}")
    if lm_studio_mode:
        tags = get_chat_completion_lmstudio(prompt, model)
    else:
        tags = get_chat_completion(prompt, model)

    # now ask it to format it as json
    prompt = f"Please format the following list of tags into a JSON list format. Only print the tags in the JSON list, do not explain it, do not make it a dictionary. Here is an example of the format: ['tag 1', 'tag 2'] Here are the tags: \"{tags}\""
    # print(f"Sending cleanup prompt")
    # note we can do this with a smaller model 
    # if lm_studio_mode:
    #     tags_raw = get_chat_completion_lmstudio(prompt, model)
    # else:
    #     tags_raw = get_chat_completion(prompt, model)
    tags_raw = get_chat_completion(prompt, model="llama3.2:latest")

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
    response = ollama.embeddings(model="mxbai-embed-large", prompt=text, keep_alive=1)
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
    result = ollama.generate(model=model, prompt=prompt, keep_alive=1)
    assert "response" in result.keys(), f"ollama response looks bad {result}"
    description = result["response"]
    # result = ollama.chat(model=model, messages=prompt, keep_alive=1)
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
        tag_embeddings[row['tag']] = np.array(json.loads(row[embedding_field]))
    return tag_embeddings
    

def get_pca_variances(embeddings):
    """
    returns a dictionary mapping number of PCA components
    to explained variance, e.g. x[100] = 0.95 # 100 components explains 95% of variance
    """
    maxn = len(embeddings)
    pca_components = np.arange(2, maxn, maxn/100) # 10 values from 2 to max components
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