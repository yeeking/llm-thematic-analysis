### This script takes a set of tags and refines them
### by reducing repetition based on semantic similarity
### i.e. if two tags are excessively semantically similar, 
### they are merged into a single tag
### The definition of 'excessively semantically similar' 
### is based on the distribution of similarities in the tag dataset
### so if the semantic distance between two tags is statistically
### significantly low (z-score) then they are merged 

