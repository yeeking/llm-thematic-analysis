
## This script composes the tags assigned in step 2 
## into themes. 
## essentially it carries out a clustering process
## on the tags seeking optimal number of clusters
## based on the classic methods of centroid distance and low overlap

# a) cluster the tags by cosine distance, optimising for 
# spread and overlap as is the standard method for clustering 
# b) According to Braun this process is focused on the codes 
# as opposed to the extracts, i.e. we are not digging back into 
# the text extracts attached to the codes here 
# c) Can also combine codes into sub themes here and REJECT unwanted codes 