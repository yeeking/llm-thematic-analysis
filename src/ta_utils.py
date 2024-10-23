import requests 
import os 
import json
from nltk.tokenize import TextTilingTokenizer
import ast 

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
API_TOKEN = "sk-7f60c0813c8f4f3ba5aa9db99365de97" # wispa
 
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


def get_chat_completion(prompt:str, max_tokens=100):
    """
    generic function to do a chat completion
    """
    global BASE_URL
    url = f'{BASE_URL}/api/chat/completions'
    headers = get_api_headers()
    data = {
    #   "model": "llama3.2:latest",
      "model":"llama3.1:latest", 
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
    assert ("data" in response.keys()) and ("file_ids" in response["data"].keys()), f"Collection data has no files"
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

def split_text_semantic(text:str):
    tt_tokenizer = TextTilingTokenizer()
    segments = tt_tokenizer.tokenize(text)
    return segments




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

def generate_tags(text:str, bad_tags_file='bad_tags.txt'):
    """
    generate a list of tags for the sent text
    """
    prompt = f"The following text is a an extract from an interview. Here is the text: \"{text}\". I would like you to generate some tags which describe the text. The tags can have one, two or three words and should describe the text and also identify the intention, sentiment or emotional content of the text. An example of such a tag is: \"happy about the weather\".  You do not need to explain the tags, just print out the list of tags."

    tags = get_chat_completion(prompt)

    # now ask it to format it as json
    prompt = f"Please format the following list of tags into a JSON list format. Only print the tags in the JSON list, do not explain it, do not make it a dictioary. Here is an example of the format: ['tag 1', 'tag 2'] Here are the tags: \"{tags}\""
    tags_raw = get_chat_completion(prompt)
    print(f"\n\n***Raw tag data: {tags_raw}")
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