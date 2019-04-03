import os
import re
import numpy as np
import networkx as nx
from time import time
from node2vec import Graph

# = = = = = = = = = = = = = = = 
# 'atoi' and 'natural_keys' taken from: https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]
# = = = = = = = = = = = = = = =
    
# = = = = = = = = = = = = = = = 
# documents padding to match maximum words by sentence and sentences by doc
def pad_docs(docs, sent_max, l_max):
    # sentence padding
    docs = [d+[[pad_vec_idx]*(l_max)]*(sent_max-len(d)) if len(d)<sent_max else d[:sent_max] for d in docs]
    for d in docs:
        for s in d:
            if len(s) < l_max:
                s += [pad_vec_idx]*(l_max-len(s))
    return docs
# = = = = = = = = = = = = = = =

pad_vec_idx = 1685894 # 0-based index of the last row of the embedding matrix (for zero-padding)

# = = = = = = = = = = = = = = =
### Parameters and paths
sent_max = 160

l_max = 11
l_i = (8, 11) # interval for random sentences length

r = 8         # nb of sentences by word
p = 2         # return hyperparam
q = 2       # in-out hyperparam

path_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath('').replace('\\', '/'))))
path_to_data = path_root + '/data/'
# = = = = = = = = = = = = = = =

def main():

    start_time = time() 

    edgelists = os.listdir(path_to_data + 'edge_lists/')
    edgelists.sort(key=natural_keys) # important to maintain alignment with the targets!
    docs = []
    for idx, edgelist in enumerate(edgelists):
        # construct graph from edgelist
        g = Graph(nx.read_edgelist(path_to_data + 'edge_lists/' + edgelist), p, q) 
        g.preprocess_transition_probs()
        
        doc = g.simulate_walks(r, l_i)
        
        # create the pseudo-document representation of the graph
        docs.append(doc)
        
        # progress tracking
        if idx % 15000 == 0:
            print(idx)

    print('documents generated')
    
    # truncation-padding at the document level, i.e., adding or removing entire 'sentences'
    docs = pad_docs(docs, sent_max, l_max)
    
    # docs = [d+[[pad_vec_idx]*(walk_length+1)]*(max_doc_size-len(d)) if len(d)<max_doc_size else d[:max_doc_size] for d in docs] 

    docs = np.array(docs).astype('int')
    print('document array shape:', docs.shape)

    np.save(path_to_data + 'documents.npy', docs, allow_pickle=False)

    print('documents saved')
    print('everything done in', str(round(time() - start_time, 2)) + 's')

# = = = = = = = = = = = = = = =

if __name__ == '__main__':
    main()
