import hashlib 
import sys
import os 
from numpy import dot
from numpy.linalg import norm
import requests

## Workflow for thematic analysis
## 1: familarise with data through summarisation
## 2: iteratively attach codes to data fragments
## 3: Search for Themes (Pattern Identification)  


BASE_URL = "http://localhost:5000"  # Replace with your Open WebUI instance URL

def api_call(endpoint: str, payload: dict):
    url = f"{BASE_URL}{endpoint}"
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json()  # Parse JSON response
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None


def str_to_hash(input:str):
    """
    one-way convert the sent string to a hash so I can use it as a key
    """
    return hashlib.sha256(input.encode()).hexdigest()[:20]  

def cosine_distance(emb1:list, emb2:list):
    """
    compute the cosine distance between the two embeddings vectors
    """
    return 1 - dot(emb1, emb2) / (norm(emb1) * norm(emb2))

def str_to_embs(input: str):
    """
    Convert the sent string to embeddings
    """
    payload = {
        "input": input
    }
    response = api_call('/api/embeddings', payload)
    
    if response:
        return response['embeddings']
    return None


def get_all_doc_frags_from_datastore(datatore_name:str):
    """
    returm a list of all document fragments from the datastore 
    """
    pass

def get_all_docs_as_strings_from_datastore(datatore_name:str):
    """
    returns a list of strings where each string is a document from the document store
    """
    pass


def add_file_to_db(token, file_path):
    url = 'http://localhost:8080/api/v1/files/'
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json'
    }
    files = {'file': open(file_path, 'rb')}
    response = requests.post(url, headers=headers, files=files)
    return response.json()

def create_knowledge(token, knowledge_title):
    url = 'http://localhost:8080/api/v1/knowledge/create'
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json'
    }
    data = {
        "name": knowledge_title,
        "description": "string",
        "data": {}
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def add_file_to_knowledge(token, file_id, knowledge_id):
    url = f'http://localhost:8080/api/v1/knowledge/{knowledge_id}/file/add'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    data = {'file_id': file_id}
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def s0_load_docs_to_knowledge(file_list: list, knowledge_name: str, token: str):
    """
    Iterate over the file list and upload them into the RAG datastore 'datastore_name'
    """

    # create the data store
    print(f"Creating knowledge store {knowledge_name}")
    knowledge_res = create_knowledge(token, knowledge_name)
    knowledge_id = knowledge_res["id"]

    # upload files
    file_ids = []
    for f in file_list:
        print(f"Adding file {f}")
        file_res = add_file_to_db(token=token, file_path=f)
        file_id = file_res["id"]
        file_ids.append(file_id)
        # insert the docs to the ks
        add_file_to_knowledge(file_id=file_id, token=token, knowledge_id=knowledge_id)
    return knowledge_id, file_ids


def get_document_contents(token, file_id):
    """
    return the contents of the document with the sent id
    """
    url = f'http://localhost:8080/api/v1/files/{file_id}/data/content'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    response = requests.get(url, headers=headers)

    return response.json()["content"]


def s1_summarise_document(token:str, docstring: str):
    """
`   instruct the LLM to generate a summary of the sent document
    """
    url = f'http://localhost:8080/api/chat/completions'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    data = {
      "model": "llama3.2:latest",
      "messages": [
        {
          "role": "user",
          "content": f"Please generate a summary of the following document\n\n{docstring}"
        }
      ]
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()
    

def s2b_filter_codes(new_codes: list, existing_code_embs: dict, similarity_threshold: float = 0.9):
    """
    remove codes from new_codes that are excessively similar to 
    items in existing_code_embs.keys() and swap them for the key from existing_code_enbs
    use cosine distance in embedding space so convert codes to embedding space 
    similar means being below similarity threshold
    new_codes: list of new codes
    existing_code_embs: dict mapping code to embeddings
    return items from new_codes that are not excessively similar, as well as excessively similar matches from existing_code_embs
    """
    filtered_codes = []
    for code in new_codes:
        code_emb = str_to_embs(code)
        similar_found = False
        for existing_code, existing_emb in existing_code_embs.items():
            if cosine_distance(code_emb, existing_emb) < similarity_threshold:
                similar_found = True
                filtered_codes.append(existing_code)
                break
        if not similar_found:
            filtered_codes.append(code)
    return filtered_codes

def s2a_generate_codes(doc_fragment: str):
    """
    Generate new codes for the sent document fragment
    Return generated codes
    """
    payload = {
        "prompt": f"Generate concise codes or labels for this fragment: {doc_fragment}",
        "max_tokens": 50  # Limit to concise responses
    }
    response = api_call('/api/chat/completions', payload)
    
    if response:
        return response['choices'][0]['text'].splitlines()
    return []



def s2_attach_codes(doc_fragments:list):
    """
    Iterate over the sent document fragments. 
    generate codes for each fragment
    each time a code is generated, check similarity to pre-existing codes
    and reject the new code if it is excessively similar to existing codes
    """
    codes_to_embs = {}
    frags_to_codes = {}
    for frag in doc_fragments:
        new_codes = s2a_generate_codes(frag)
        # this will return the codes from new_codes 
        # that are not excessively similar to existing codes
        # and replace the very similar ones with existing ones from existing_codes
        filtered_codes = s2b_filter_codes(new_codes, codes_to_embs)
        # compute embeddings for filtered_codes and add to existing_codes
        for code in filtered_codes:
            emb = str_to_embs(code)
            codes_to_embs[code] = emb
        # attach codes to this document
        frag_key = str_to_hash(frag)
        assert frag_key not in frags_to_codes.keys(), "Bad key!"
        frags_to_codes[frag_key] = filtered_codes
    
    return frags_to_codes, codes_to_embs


## 3: Search for Themes (Pattern Identification)  
def s3_codes_to_themes(doc_frags:list, frags_to_codes:dict, codes_to_embs:dict):
    """
    get codes_to_embs.keys() which is the complete list of codes
    cluster the codes based on semantic distance between their embeddings
    then ask the LLM to come up with a descriptive theme title for each cluster 
    """
    # this will map theme descriptions to lists of codes
    themes_to_codes = {}

    pass

def get_files_in_folder(folder: str):
    """
    Return a list of file paths in the sent folder
    """
    return [os.path.join(folder, file) for file in os.listdir(folder) if os.path.isfile(os.path.join(folder, file))]


if __name__ == "__main__":
    print("Here we go!")
    assert len(sys.argv) == 2, "Usage: python script.py document_folder"
    folder = sys.argv[1]
    assert os.path.exists(folder), f"Imput Folder not found {folder}"

    file_list = get_files_in_folder(folder)
    knowledge_name = "my_docs"
    token = "sk-43130b6612624d6aaaecb5fa980fda0c"

    knowledge_id, file_ids = s0_load_docs_to_knowledge(file_list=file_list, knowledge_name=knowledge_name, token=token)
    for fid in file_ids:
        print(f"Analysing {fid}")
        text = get_document_contents(token, fid)
        print(text[0:10])
        summ = s1_summarise_document(token, text)
        print("Got summary")
        print(summ)

    # s0_load_docs_to_datastore(file_list, datastore_name, 'sk-43130b6612624d6aaaecb5fa980fda0c')
    # doc_strings = get_all_docs_as_strings_from_datastore(datastore_name)
    # summaries = s1_summarise_document(doc_strings)
    # frags = get_all_doc_frags_from_datastore(datastore_name)
    # frags_to_codes, codes_to_embs = s2_attach_codes(frags)
    # themes = s3_codes_to_themes(frags, frags_to_codes, codes_to_embs)


