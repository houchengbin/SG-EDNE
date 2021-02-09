"""
Skip-Gram based Ensembles Dynamic Network Embedding with fixed Random Walks (SG-EDNE-fixRW)
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



class SG_EDNE_fixRW(object):
    def __init__(self, G_dynamic, num_walks, walk_length, emb_dim, window, negative, workers, seed,
                    num_base_models, sampling_strategy, combining_strategy, scaling_strategy):
        self.G_dynamic = G_dynamic.copy()   # a series of dynamic graphs
        self.num_walks = num_walks          # num of walks start from each node
        self.walk_length = walk_length      # walk length for each walk
        self.emb_dim = emb_dim              # node emb dimensionarity
        self.window = window                # gensim word2vec parameters
        self.negative = negative            # gensim word2vec parameters
        self.seed = seed                    # gensim word2vec parameters
        self.workers = workers              # gensim word2vec parameters
        self.num_base_models = num_base_models
        self.sampling_strategy = sampling_strategy
        self.combining_strategy = combining_strategy
        self.scaling_strategy = scaling_strategy
        self.emb_dicts = [] # emb_dict @ t0, t1, ...; len(self.emb_dicts) == len(self.G_dynamic)

        if self.emb_dim < self.num_base_models:
            exit('Exit, please make sure emb_dim >= num_base_models')
        else:
            remainder = self.emb_dim % self.num_base_models
            if remainder != 0:
                print('num_base_models', self.num_base_models)
                print('final emb_dim', self.emb_dim)
                print('original combining_strategy', self.combining_strategy)
                print('***Warning*** force to choice combining_strategy=1 i.e. concat, as emb_dim is not divisible by num_base_models')
                self.combining_strategy = 1
                self.emb_dim = [self.emb_dim//self.num_base_models]*self.num_base_models
                self.emb_dim[-1] = self.emb_dim[-1] + remainder
                print('new combining_strategy', self.combining_strategy)
                print('emb_dim for each base model', self.emb_dim)
            else:
                print('final emb_dim', self.emb_dim)
                self.emb_dim = [self.emb_dim//self.num_base_models]*self.num_base_models
                print('emb_dim for each base model', self.emb_dim)
                print('combining_strategy', self.combining_strategy)


    def traning(self):
        # -------- initialize base models for ensemble --------
        w2v_models = []
        for i in range(self.num_base_models):
            # we can tune vector_size, window, negative, workers, and seed from CMD, e.g., python main.py --parameter1 x1 --parametere2 x2 ...
            # other parameters follow https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec (gensim==4.0.0b0 is used)
            # by default, early stop for 5 epochs
            w2v_models.append(gensim.models.word2vec.Word2Vec(sentences=None, vector_size=self.emb_dim[i], window=self.window, negative=self.negative,
                            workers=self.workers, seed=self.seed, sg=1, hs=0, ns_exponent=0.75, epochs=5, batch_words=10000, sample=0.001, alpha=0.025,
                            min_alpha=0.0001, min_count=5, sorted_vocab=1, compute_loss=False, max_vocab_size=None, max_final_vocab=None))

        # -------- sampling, traning, combining, scaling --------
        for t in range(len(self.G_dynamic)):
            t1 = time.time()
            if t == 0: # initial training
                # fix RW for all base models, to test whether diversity is useful
                sentences = simulate_walks(nx_graph=self.G_dynamic[0], num_walks=self.num_walks, walk_length=self.walk_length, start_node=None)
                sentences = [[str(j) for j in i] for i in sentences]
                for i in range(self.num_base_models):
                    print('repeat training fixed RW @ ', i)
                    w2v_models[i].build_vocab(corpus_iterable=sentences, update=False)
                    w2v_models[i].train(corpus_iterable=sentences, total_examples=w2v_models[i].corpus_count, epochs=w2v_models[i].epochs)
            else: # incremental training
                # ---- sampling ----
                node_samples, node_add, node_del = self.sampling(graph_t0=self.G_dynamic[t-1], graph_t1=self.G_dynamic[t], sampling_strategy=self.sampling_strategy, emb_dict_t0=self.emb_dicts[-1])
                node_sample = node_samples[0] # fix nodes, make sure node_samples[0] = node_samples[1]
                # fix RW for all base models, to test whether diversity is useful
                sentences = simulate_walks(nx_graph=self.G_dynamic[t], num_walks=self.num_walks, walk_length=self.walk_length, start_node=node_sample) # RW
                sentences = [[str(j) for j in i] for i in sentences] # fixed RW for all models
                for node in node_add: # to ensure all unseen nodes are included
                    sentences.append([str(node)]*self.window)
                # ---- traning ----
                for i in range(self.num_base_models): # for each model, the sentences (fixed RW) are exactly the same!
                    print('repeat training fixed RW @ ', i)
                    w2v_models[i].build_vocab(corpus_iterable=sentences, update=True)
                    w2v_models[i].train(corpus_iterable=sentences, total_examples=w2v_models[i].corpus_count, epochs=w2v_models[i].epochs)
            # ---- combining ----
            emb_dict = self.combining(nx_graph=self.G_dynamic[t], combining_strategy=self.combining_strategy, w2v_models=w2v_models)
            # ---- scaling ----
            emb_dict = self.scaling(emb_dict=emb_dict, scaling_strategy=self.scaling_strategy)

            # ---- return all embeddings for all time steps ----
            self.emb_dicts.append(emb_dict)
            print('final emb_dim', len(list(emb_dict.values())[0]))
            t2 = time.time()
            print(f'@t{t}, total traning time cost: {(t2-t1):.2f}s --> {t+1}/{len(self.G_dynamic)}')
        return self.emb_dicts


    def sampling(self, graph_t0, graph_t1, sampling_strategy, emb_dict_t0=None, w2v_models=None):
        ''' sampling strategies
        1: naively repeat all affected nodes
        2: sample with replancement with equal probability
        3: sample with replancement with equal probability; partially e.g. 0.8
        '''
        t1 = time.time()
        G0 = graph_t0.copy()
        G1 = graph_t1.copy()
        edge_add = edge_s1_minus_s0(s1=set(G1.edges()), s0=set(G0.edges()))  # one may directly use edge streams if available
        edge_del = edge_s1_minus_s0(s1=set(G0.edges()), s0=set(G1.edges()))
        node_affected_by_edge_add = unique_nodes_from_edge_set(edge_add)
        node_affected_by_edge_del = unique_nodes_from_edge_set(edge_del)
        node_affected = list(set(node_affected_by_edge_add + node_affected_by_edge_del))  # nodes being directly affected between G0 and G1
        node_add = [node for node in node_affected_by_edge_add if node not in G0.nodes()] # nodes not exist in G0
        node_del = [node for node in node_affected_by_edge_del if node not in G1.nodes()] # nodes not exist in G1
        node_sample_pool = list(set(node_affected) - set(node_del))                       # nodes being directly affected in G1
        print(f'# node being affected in current graph: {len(node_sample_pool)}')
        node_samples = [] # list of list

        # naively repeat all affected nodes
        if sampling_strategy == 1:
            print('S1: naively repeat all affected nodes')
            for i in range(self.num_base_models):
                node_samples.append(node_sample_pool)
        # sample with replancement with equal probability
        elif sampling_strategy == 2:
            print('S2: sample with replancement with equal probability')
            for i in range(self.num_base_models):
                node_samples.append(list(np.random.choice(node_sample_pool, size=len(node_sample_pool), replace=True)))
        # sample with replancement with equal probability; partially e.g. 80%
        elif sampling_strategy == 3:
            print('S3: sample with replancement with equal probability; partially e.g. 80%')
            frac = 0.80
            for i in range(self.num_base_models):
                node_samples.append(list(np.random.choice(node_sample_pool, size=int(frac*len(node_sample_pool)), replace=True)))
        else:
            exit('Exit, sampling strategy not found ...')
        t2 = time.time()
        print(f'sampling time cost: {(t2-t1):.2f}')
        return node_samples, node_add, node_del


    def combining(self, nx_graph, combining_strategy, w2v_models):
        ''' combining strategies
        1: concat
        2: elementwise-mean
        3: elementwise-min
        4: elementwise-max
        5: elementwise-sum
        '''
        t1 = time.time()
        emb_dict = {} # dictionary {node: emb_vector, ...}
        # naively concat
        if combining_strategy == 1:
            for node in nx_graph.nodes():
                temp_wv = []
                for i in range(self.num_base_models):
                    temp_wv.extend(w2v_models[i].wv[str(node)])
                emb_dict[node] = temp_wv
                #print('emb dim after combining', len(temp_wv)); exit(0)
        # element-wise mean
        elif combining_strategy == 2:
            for node in nx_graph.nodes():
                temp_wv = []
                for i in range(self.num_base_models):
                    temp_wv.append(w2v_models[i].wv[str(node)])
                temp_wv = np.mean(temp_wv, axis=0)
                emb_dict[node] = temp_wv
                #print('emb dim after combining', len(temp_wv)); exit(0)
        # element-wise sum
        elif combining_strategy == 3:
            for node in nx_graph.nodes():
                temp_wv = []
                for i in range(self.num_base_models):
                    temp_wv.append(w2v_models[i].wv[str(node)])
                temp_wv = np.sum(temp_wv, axis=0)
                emb_dict[node] = temp_wv
        # element-wise min
        elif combining_strategy == 4:
            for node in nx_graph.nodes():
                temp_wv = []
                for i in range(self.num_base_models):
                    temp_wv.append(w2v_models[i].wv[str(node)])
                temp_wv = np.min(temp_wv, axis=0)
                emb_dict[node] = temp_wv
        # element-wise max
        elif combining_strategy == 5:
            for node in nx_graph.nodes():
                temp_wv = []
                for i in range(self.num_base_models):
                    temp_wv.append(w2v_models[i].wv[str(node)])
                temp_wv = np.max(temp_wv, axis=0)
                emb_dict[node] = temp_wv
        else:
            exit('Exit, combining strategy not found ...')
        t2 = time.time()
        print(f'combining time cost: {(t2-t1):.2f}')
        return emb_dict


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

# =================================================================================================
# ========================================== random walk sampling =================================
# =================================================================================================
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
            cur = start_node # restart
            walk.append(cur)
        else:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
    #print('walk', walk)
    #exit(0)
    return walk