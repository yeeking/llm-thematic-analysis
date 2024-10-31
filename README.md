# llm-thematic-analysis
Workflows to use LLM for thematic and qualitative analysis


## Setup the servers

The scripts rely on two 'servers': 

* open-webui: https://docs.openwebui.com/
  This is a web interface and REST API interface on a load of LLM functionality such as RAG-able document stores, unified interface on LLM servers, chat etc.
  I use this to create the document store and to issue 'chat completion' queries

* ollama: https://ollama.com/
  This is an LLM and embeddings server. open-webui talks to this but you can also talk to it directly. 
  It makes it easy to download and serve a variety of models such as llama 3.1, 3.2 etc.


```
# ollama can be installed in various ways - choose your preferred one from the site
# I prefer to download the binaries and put them where I want as I found the
# installers to be a bit agressive with my system files

# open-webui can be installed via pip
python3 -m venv ~/Python/open-webui
source ~/Python/open-webui/bin/activate
pip install open-webui ollama ipython notebooks scipy numpy # etc
```

Once they are installed, fire them up:

```
ollama serve &
open-webui serve &
```
Then get some models installed in ollama

```
# for starters
ollama pull mxbai-embed-large:latest
ollama pull lama3.2:latest
```

Then go onto the web interface for open-webui and create a user account. The first account you create becomes the admin. 

Then login and go into your account settings and generate an API key. You'll need that later

## Prepare your data

Put some docx files in a folder. The scripts will pull those in

## Run the scripts

### Import docs to doc store

This will create a document store on the open-webui server containing the docx files in ../data/mydocs. 
The store will be called my_collection_name

```
python 0_import.py ../data/mydocs/ my_collection_name
```
If you want to delete that doc store and start again:

```
python 0_delete_collection.py my_collection_name
```

### Segment the docs and generate tags

Segment each document in collection_name (frag len and frag_hop not used) write to json_outfile. Use model to generate the tags. 

Examples of models:

```
ollama list 

NAME                     	ID          	SIZE  	MODIFIED     
llama3.1:8b              	42182419e950	4.7 GB	3 hours ago 	
llama3.2:3b-instruct-q8_0	e410b836fe61	3.4 GB	3 hours ago 	
llama3.1:70b             	c0df3564cfe8	39 GB 	8 days ago  	
llama3.1:latest          	42182419e950	4.7 GB	8 days ago  	
llama3.2:latest          	a80c4f17acd5	2.0 GB	8 days ago  	
mxbai-embed-large:latest 	468836162de7	669 MB	2 months ago	
llama3:70b               	786f3184aec0	39 GB 	5 months ago	
llama3:latest            	365c0bd3c000	4.7 GB	5 months ago
```

```
python 2a_tags.py collection_name frag_len frag_hop json_outfile model
```

### Extract embeddings for the tags

This will convert each 'tag' into a description, then extract the embeddings of the description:

```
python 2b_extract_embeddings.py json_tags_to_quotes_File csv_embeddings_file llm-model
```

