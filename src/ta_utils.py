import requests 
import os 

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
API_TOKEN = "sk-43130b6612624d6aaaecb5fa980fda0c"
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
      "model": "llama3.2:latest",
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
        collection_id = ta_utils.get_collection_id(coll_name)
        print("deleting ", collection_id)
        delete_collection(collection_id)

def does_collection_id_exist(id:str):
    """
    check if a collection exists.
    @return True or False
    """
    
    return False

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
    return response 
    


def add_doc_to_collection(collection_id, doc_contents):
    """
    create a doc in the sent collection and return its id
    """
    doc_id = ""
    return doc_id

def get_docs_in_collection(collection_id:str):
    """
    get a list of doc ids from the sent collection id
    """
    return []

def get_doc_contents(collection_id, doc_id):
    """
    get the doc contents as a string
    """
    global API_TOKEN, BASE_URL
    url = f'{BASE_URL}/api/v1/files/{doc_id}/data/content'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    response = requests.get(url, headers=headers)

    return response.json()["content"]


def summarise_text(text:str):
    """
    return a summary of the sent doc
    """
    return ""

def split_text(text:str, length:int, overlap:int):
    """
    split the sent text into chunks of the sent length with the sent overlap 
    might want to use the webui api to do this or to pull frags out of the collection
    """
    return []

def generate_tags(text:str):
    """
    generate a list of tags for the sent text
    """
    return []