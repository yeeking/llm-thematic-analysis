import requests 
import os 

## utiltiy functions for thematic analysis

## in open webui this is not super secret
## so i'm putting it here. If we talk to other apis
## this should go in the bash envt. 
API_TOKEN = "123"
BASE_URL = "http://localhost:8080/" # Replace with your Open WebUI instance URL


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
    global API_TOKEN, BASE_URL
    url = f'{BASE_URL}/api/chat/completions'
    headers = {
        'Authorization': f'Bearer {API_TOKEN}',
        'Content-Type': 'application/json'
    }
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
    text = data["choices"][0]["message"]["content"]
    return text


def list_collections():
    """
    returns a list of doc collections currently on the server
    """
    doc_collections = []
    return doc_collections

def does_collection_name_exist(name:str):
    """
    check if a collection exists.
    @return True or False
    """
    
    return False

def does_collection_id_exist(id:str):
    """
    check if a collection exists.
    @return True or False
    """
    
    return False

def create_collection(name:str):
    """
    create a collection and return its id
    """
    collection_id = ""

    return collection_id


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
    return ""


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