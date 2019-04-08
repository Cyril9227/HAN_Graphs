import numpy as np
import random
# fix seed for reproducibility
np.random.seed(2019)

class Graph():
    '''
    Graph class: wrapper for an input graph, provides methods to perform
    biased random walks (analogous to node2vec)
    https://github.com/aditya-grover/node2vec/blob/master/src/node2vec.py
    '''
    def __init__(self, nx_G, p, q):
        self.G = nx_G
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        # Initial node
        walk = [start_node]
        while len(walk) < walk_length:
            # Current node in the walk
            cur = walk[-1]
            # Current node's neighbours
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                # If we just started, we don't have any preceding node
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                # Else the probability distribution is conditioned on the preceding node
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], 
                                                           alias_edges[(prev, cur)][1])]
                    walk.append(next)
            # An isolated node will always form sentences on its own
            else:
                break
            
        return walk

    def simulate_walks(self, num_walks, law, length_d):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        # Repeat num_walks times
        for walk_iter in range(num_walks):
            # For each node (not always the same first, random order)
            random.shuffle(nodes)
            for node in nodes:
                # The length of the walk follows the provided distribution
                dist = random.randint if law == 'uniform' else random.normalvariate
                walk_length = int(dist(length_d[0], length_d[1]))
                # Perform from the node a random walk of given length, biased by pi
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
                
        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(1/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(1)
            else:
                unnormalized_probs.append(1/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
        
        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [1 for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}

        for edge in G.edges():
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
            alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return

def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    => Permits O(1) random draws after this setup
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
    
    