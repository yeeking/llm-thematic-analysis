re-running the analysis:

## Start services

Start up the various services, in my case:

* Run LM-studio, start API with gemma 27b model
* open-webui serve & 
* ollama serve &

## Import docs

Import docs from folder into collection:

```
python 0a_import.py ../docs/split/exam_setter/ exam-setter-v2
```

## Generate tags

The dataset, frag length and frag overlap and model (model is ignored if lm-studio - that just uses whatever the API is currently serving). 

```
python 2a_tags.py exam-setter-v2 3 1 gemma27b
```

That generates a file called 'examsetterv2gemma27b.json' (removes non-alphanumeric characters from the dataset name and model name and mashes them together)



## Now can run the whole deal with a single script I think

Edit 2b_to_3b.py, make sure the models and datasets are correct:

```
# started like this:
datasets = ["collusionmac", "examsettermac"]
models = ["gemma27b", "llama323b", "llama3170b"]
# I'm only running one model, one dataset so I changed it to:
datasets = ["exam-setter-v2"]
models = ["gemma27b"]
```

Then run it: 

```
python 2b_to_3b.py ../data/ 2b,2c.3a.3b,4a,4b,4d
```

It generates a load of files, including the tags + themes cluster visualisation that goes into the paper


## Generate the heatmap

```
python 4g_dataset_theme_distance_plot.py 
```














