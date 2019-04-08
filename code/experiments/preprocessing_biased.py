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

def pad_docs(docs, sent_max, l_max):
    '''
    documents padding to match maximum words by sentence and sentences by doc
    sent_max: maximum number of sentences by documents
    l_max: maximum number of words by sentence
    '''
    # sentence padding
    docs = [d+[[pad_vec_idx]*(l_max)]*(sent_max-len(d)) if len(d)<sent_max else d[:sent_max] for d in docs]
    # word padding
    for d in docs:
        for i in range(len(d)):
            d[i] = d[i] + [pad_vec_idx]*(l_max-len(d[i])) if len(d[i]) < l_max else d[i][:l_max]
    return docs
# = = = = = = = = = = = = = = =
    
pad_vec_idx = 1685894 # 0-based index of the last row of the embedding matrix (for zero-padding)

# = = = = = = = = = = = = = = =
### Paths
path_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath('').replace('\\', '/'))))
path_to_data = path_root + '/data/'
# = = = = = = = = = = = = = = =

def create_documents(target, sent_max, sent_l_law, sent_l_stats, word_rate, p, q):
    '''
    target: 0 to 3: number of the target for which we want to use the embedded docs
    sent_max: maximum number of sentences in produced documents
    sent_l_law: 'uniform' or 'normal' => the distribution to follow when generating random sentence length
    sent_l_stats: the interval of lengths if uniform, mean and std if normal distribution
    word_rate (r): number of sentences that start from each word/node
    p: return parameter (controls how often we stay at the same node within a walk)
    q: in-out parameter (q << 1 => outward walks, DFS exploration, q >> 1 => inward walk, BFS exploration)
    '''
    if sent_l_law != 'uniform' and sent_l_law != 'normal':
        return
    start_time = time() 

    edgelists = os.listdir(path_to_data + 'edge_lists/')
    edgelists.sort(key=natural_keys) # important to maintain alignment with the targets!
    docs = []
    for idx, edgelist in enumerate(edgelists):
        # construct graph from edgelist
        g = Graph(nx.read_edgelist(path_to_data + 'edge_lists/' + edgelist), p, q) 
        g.preprocess_transition_probs()
        
        # get document as a biased random walk (similar to node2vec sampling)
        doc = g.simulate_walks(word_rate, sent_l_law, sent_l_stats)
        
        # create the pseudo-document representation of the graph
        docs.append(doc)
        
        # progress tracking
        if idx % 15000 == 0:
            print(idx)

    print('documents generated for target', target)
    
    # truncation-padding at the document level, i.e., adding or removing entire 'sentences'
    word_max = sent_l_stats[1] if sent_l_law == 'uniform' else int(sent_l_stats[0] + 1.5 * sent_l_stats[1])
    docs = pad_docs(docs, sent_max, word_max)

    docs = np.array(docs).astype('int')
    print('document array shape:', docs.shape)

    np.save(path_to_data + 'documents_' + str(target) + '.npy', docs, allow_pickle=False)

    print('documents saved for target', target)
    print('everything done in', str(round(time() - start_time, 2)) + 's')
