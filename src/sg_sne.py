"""
Skip-Gram based Static Network Embedding at each time step (SG-SNE)
"""

import time
import random
import pickle
import gensim
#import warnings
#warnings.filterwarnings("ignore")
#warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import numpy as np
import networkx as nx
#import math
from utils import edge_s1_minus_s0, unique_nodes_from_edge_set



class SG_SNE(object):
    def __init__(self, G_dynamic, num_walks, walk_length, emb_dim, window, negative, workers, seed, scaling_strategy):
        self.G_dynamic = G_dynamic.copy()   # a series of dynamic graphs
        self.num_walks = num_walks          # num of walks start from each node
        self.walk_length = walk_length      # walk length for each walk
        self.emb_dim = emb_dim              # node emb dimensionarity
        self.window = window                # gensim word2vec parameters
        self.negative = negative            # gensim word2vec parameters
        self.seed = seed                    # gensim word2vec parameters
        self.workers = workers              # gensim word2vec parameters
        self.scaling_strategy = scaling_strategy
        self.emb_dicts = [] # emb_dict @ t0, t1, ...; len(self.emb_dicts) == len(self.G_dynamic)

    def traning(self):
        for t in range(len(self.G_dynamic)):
            t1 = time.time()
            w2v = gensim.models.word2vec.Word2Vec(sentences=None, vector_size=self.emb_dim, window=self.window, negative=self.negative,
                            workers=self.workers, seed=self.seed, sg=1, hs=0, ns_exponent=0.75, epochs=5, batch_words=10000, sample=0.001, alpha=0.025,
                            min_alpha=0.0001, min_count=5, sorted_vocab=1, compute_loss=False, max_vocab_size=None, max_final_vocab=None) 
            sentences = simulate_walks(nx_graph=self.G_dynamic[t], num_walks=self.num_walks, walk_length=self.walk_length, start_node=None)
            sentences = [[str(j) for j in i] for i in sentences]
            w2v.build_vocab(corpus_iterable=sentences, update=False)
            w2v.train(corpus_iterable=sentences, total_examples=w2v.corpus_count, epochs=w2v.epochs) # follow w2v constructor

            emb_dict = {} # {nodeID: emb_vector, ...}
            for node in self.G_dynamic[t].nodes():
                emb_dict[node] = w2v.wv[str(node)]
                
            emb_dict = self.scaling(emb_dict=emb_dict, scaling_strategy=self.scaling_strategy)
            self.emb_dicts.append(emb_dict)
            t2 = time.time()
            print(f'Static NE at each timestep; DeepWalk-like; traning time: {(t2-t1):.2f}s --> {t+1}/{len(self.G_dynamic)} graphs')
            del w2v
        return self.emb_dicts 

    def scaling(self, emb_dict, scaling_strategy):
        '''
        scaling strategies
        0: do nothing
        1: MinMaxScaler [0~1]
        2: MinMaxScaler [-1~1]
        3: StandardScaler, 0 mean, 1 var
        https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
        '''
        # do nothing
        if scaling_strategy == 0:
            print(f'scaling emb, do nothing!')
        # MinMaxScaler [0,1]
        elif scaling_strategy == 1:
            t_emb_norm_0 = time.time()
            from sklearn.preprocessing import MinMaxScaler
            data = list(emb_dict.values())
            transforred_data = MinMaxScaler(feature_range=(0,1)).fit_transform(data)
            #for i in range(len(transforred_data[0,:])): # along each feat
            #    print('min ', np.min(transforred_data[:,i]))
            #    print('max', np.max(transforred_data[:,i]))
            #exit(0)
            data_keys = list(emb_dict.keys())
            for i in range(len(data_keys)):
                emb_dict[data_keys[i]] = transforred_data[i]
            t_emb_norm_1 = time.time()
            print(f'emb MinMaxScaler (0,1) time cost: {(t_emb_norm_1-t_emb_norm_0):.2f}s')
        # MinMaxScaler [-1,1]
        elif scaling_strategy == 2:
            t_emb_norm_0 = time.time()
            from sklearn.preprocessing import MinMaxScaler
            data = list(emb_dict.values())
            transforred_data = MinMaxScaler(feature_range=(-1, 1)).fit_transform(data)
            data_keys = list(emb_dict.keys())
            for i in range(len(data_keys)):
                emb_dict[data_keys[i]] = transforred_data[i]
            t_emb_norm_1 = time.time()
            print(f'emb MinMaxScaler (-1, 1) time cost: {(t_emb_norm_1-t_emb_norm_0):.2f}s')
        # StandardScaler zero mean unit var
        elif scaling_strategy == 3:
            t_emb_norm_0 = time.time()
            from sklearn.preprocessing import StandardScaler
            data = list(emb_dict.values())
            transforred_data = StandardScaler().fit_transform(data)
            #for i in range(len(transforred_data[0,:])): # along each feat
            #    print('mean ',np.mean(transforred_data[:,i]))
            #    print('var ',np.var(transforred_data[:,i]))
            #exit(0)
            data_keys = list(emb_dict.keys())
            for i in range(len(data_keys)):
                emb_dict[data_keys[i]] = transforred_data[i]
            t_emb_norm_1 = time.time()
            print(f'emb StandardScaler time cost: {(t_emb_norm_1-t_emb_norm_0):.2f}s')
        else:
            exit('Exit, scaling strategy not found ...')
        return emb_dict

    def save_emb(self, path='unnamed_dyn_emb_dicts.pkl'):
        ''' save # emb_dict @ t0, t1, ... to a file using pickle
        '''
        with open(path, 'wb') as f:
            pickle.dump(self.emb_dicts, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_emb(self, path='unnamed_dyn_emb_dicts.pkl'):
        ''' load # emb_dict @ t0, t1, ... to a file using pickle
        '''
        with open(path, 'rb') as f:
            any_object = pickle.load(f)
        return any_object


def simulate_walks(nx_graph, num_walks, walk_length, restart_prob=None, start_node=None):
    '''
    Repeatedly simulate random walks from each node
    '''
    t1 = time.time()
    G = nx_graph
    walks = []

    if start_node == None:   # simulate walks on every node in the graph [offline]
        nodes = list(G.nodes())
    else:                     # simulate walks on affected nodes [online]
        nodes = list(start_node)
    """ multi-processors; use it iff the # of nodes over 20k
    if restart_prob == None: # naive random walk
        t1 = time.time()
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            from itertools import repeat
            from multiprocessing import Pool, freeze_support
            with Pool(processes=5) as pool:
                # results = [pool.apply_async(random_walk, args=(G, node, walk_length)) for node in nodes]
                # results = [p.get() for p in results]
                results = pool.starmap(random_walk, zip(repeat(G), nodes, repeat(walk_length)))
            for result in results:
                walks.append(result)
        t2 = time.time()
        print('all walks',len(walks))
        print(f'random walk sampling, time cost: {(t2-t1):.2f}')
    """
    if restart_prob == None: # naive random walk
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(random_walk(nx_graph=G, start_node=node, walk_length=walk_length))
    else: # random walk with restart
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(random_walk_restart(nx_graph=G, start_node=node, walk_length=walk_length, restart_prob=restart_prob))
    t2 = time.time()
    print(f'random walk, time cost: {(t2-t1):.2f}')
    return walks

def random_walk(nx_graph, start_node, walk_length):
    '''
    Simulate a random walk starting from start node
    '''
    G = nx_graph
    walk = [start_node]

    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = list(G.neighbors(cur))
        if len(cur_nbrs) > 0:
            walk.append(random.choice(cur_nbrs))
        else:
            break
    return walk

def random_walk_restart(nx_graph, start_node, walk_length, restart_prob):
    '''
    random walk with restart
    restart if p < restart_prob
    '''
    G = nx_graph
    walk = [start_node]

    while len(walk) < walk_length:
        p = random.uniform(0, 1)
        if p < restart_prob:
            cur = walk[0] # restart
            walk.append(cur)
        else:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
    return walk
