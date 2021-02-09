"""
Skip-Gram based Ensembles Dynamic Network Embedding (SG-EDNE)
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



class SG_EDNE(object):
    def __init__(self, G_dynamic, num_walks, walk_length, emb_dim, window, negative, workers, seed,
                    num_base_models, restart, max_r_prob, sampling_strategy, combining_strategy, scaling_strategy):
        self.G_dynamic = G_dynamic.copy()   # a series of dynamic graphs
        self.num_walks = num_walks          # num of walks start from each node
        self.walk_length = walk_length      # walk length for each walk
        self.emb_dim = emb_dim              # node emb dimensionarity
        self.window = window                # gensim word2vec parameters
        self.negative = negative            # gensim word2vec parameters
        self.seed = seed                    # gensim word2vec parameters
        self.workers = workers              # gensim word2vec parameters
        self.restart = restart
        self.max_r_prob = max_r_prob
        self.num_base_models = num_base_models
        self.sampling_strategy = sampling_strategy
        self.combining_strategy = combining_strategy
        self.scaling_strategy = scaling_strategy
        self.emb_dicts = [] # emb_dict @ t0, t1, ...; len(self.emb_dicts) == len(self.G_dynamic)

        if self.restart == True:
            p = self.max_r_prob / self.num_base_models # assgin diverse diff restart prob based on max_r_prob
            self.r_prob = [i*p for i in range(self.num_base_models)]

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
                for i in range(self.num_base_models):
                    if self.restart == True:
                        print(f'@t{t}, base model {i}, restart_prob=self.r_prob[i] {self.r_prob[i]}')
                        sentences = simulate_walks(nx_graph=self.G_dynamic[0], num_walks=self.num_walks, walk_length=self.walk_length, start_node=None, restart_prob=self.r_prob[i])
                    else:
                        sentences = simulate_walks(nx_graph=self.G_dynamic[0], num_walks=self.num_walks, walk_length=self.walk_length, start_node=None, restart_prob=None)
                    sentences = [[str(j) for j in i] for i in sentences]
                    w2v_models[i].build_vocab(corpus_iterable=sentences, update=False)
                    w2v_models[i].train(corpus_iterable=sentences, total_examples=w2v_models[i].corpus_count, epochs=w2v_models[i].epochs)
            else: # incremental training
                # --- sampling ---
                node_samples, node_add, node_del = self.sampling(graph_t0=self.G_dynamic[t-1], graph_t1=self.G_dynamic[t], sampling_strategy=self.sampling_strategy, emb_dict_t0=self.emb_dicts[-1])
                # --- traning ---
                for i in range(self.num_base_models):
                    node_sample = node_samples[i]
                    print(f'@t{t}, base model {i}, len(node_sample) {len(node_sample)}')
                    if self.restart == True:
                        print(f'@t{t}, base model {i}, restart_prob=self.r_prob[i] {self.r_prob[i]}')
                        sentences = simulate_walks(nx_graph=self.G_dynamic[t], num_walks=self.num_walks, walk_length=self.walk_length, start_node=node_sample, restart_prob=self.r_prob[i])
                    else:
                        sentences = simulate_walks(nx_graph=self.G_dynamic[t], num_walks=self.num_walks, walk_length=self.walk_length, start_node=node_sample, restart_prob=None)
                    sentences = [[str(j) for j in i] for i in sentences]
                    for node in node_add: # to ensure all unseen nodes are included
                        sentences.append([str(node)]*self.window) # one node as a sentence -> almost no affect on their embeddings
                    #print('sentences[-1:]', sentences[-1])
                    w2v_models[i].build_vocab(corpus_iterable=sentences, update=True) # update=True for adding unseen samples i.e. node_add
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







# =========================================================================
# ============================== not used ... =============================
# =========================================================================
"""
def graph_enhancement(G0, G1, node_sample):
    t0=time.time()
    edge_add = list(edge_s1_minus_s0(s1=set(G1.edges()), s0=set(G0.edges())))  # one may directly use edge streams if available
    edge_del = list(edge_s1_minus_s0(s1=set(G0.edges()), s0=set(G1.edges())))
    edge_to_add = []
    edge_to_del = []
    for node in node_sample:
        for i in range(len(edge_add)):
            if node == edge_add[i][0] or node == edge_add[i][1]:
                edge_to_add.append(edge_add[i])
        for i in range(len(edge_del)):
            if node == edge_del[i][0] or node == edge_del[i][1]:
                edge_to_del.append(edge_del[i])
    G_enhanced = G0.copy()
    #print('len(G_enhanced.edges())', len(G_enhanced.edges()))
    #print('len(G0.edges())', len(G0.edges()))
    G_enhanced.add_edges_from(edge_to_add)
    G_enhanced.remove_edges_from(edge_to_del)
    print('len(G_enhanced.edges())', len(G_enhanced.edges()))
    print('len(G0.edges())', len(G0.edges()))
    t1=time.time()
    print('time for graph enhancement: ', t1-t0)
    return G_enhanced
def check_is_same_dim(vec1, vec2):
    if len(vec1) == len(vec2):
        return True
    else:
        exit('Exit, dim is not the same ...')

def select_most_affected_nodes_nbrs(G1, most_affected_nodes):
    most_affected_nbrs = []
    for node in most_affected_nodes:
        most_affected_nbrs.extend( list(nx.neighbors(G=G1, n=node)) )
    return list(set(most_affected_nbrs))

def sentences_2_pkl(my_list, path):
    import collections
    new_list = []
    for items in my_list:
        for item in items:
            new_list.append(item)
    c = collections.Counter(new_list)

    with open(path, 'wb') as f:
        pickle.dump(c, f, protocol=pickle.HIGHEST_PROTOCOL)

def node_update_list_2_txt(my_list, path):
    with open(path, 'w') as f:
        for item in my_list:
            f.write("%s\n" % item)

def to_weighted_graph(graph):
    G = graph.copy()
    for e in G.edges():
        G[e[0]][e[1]]['weight'] = 1.0
    return G

def to_unweighted_graph(weighted_graph):
    pass
    # return unweighted_graph
"""


# scaling strategies to delete...
"""
#4 RobustScaler
t_emb_norm_0 = time.time()
from sklearn.preprocessing import RobustScaler
data = list(emb_dict.values())
transforred_data = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)).fit_transform(data)
#for i in range(len(transforred_data[0,:])): # along each feat
#    print('min ', np.min(transforred_data[:,i]))
#    print('max', np.max(transforred_data[:,i]))
#exit(0)
data_keys = list(emb_dict.keys())
for i in range(len(data_keys)):
    emb_dict[data_keys[i]] = transforred_data[i]
t_emb_norm_1 = time.time()
print(f'@t{t}, emb RobustScaler (0,1)  time cost: {(t_emb_norm_1-t_emb_norm_0):.2f}s')
"""
"""
#5 L2 all in one
t_emb_norm_0 = time.time()
from sklearn.preprocessing import Normalizer
data = list(emb_dict.values())
transforred_data = Normalizer(norm='l2').fit_transform(data)
#from numpy import linalg as LA
#for i in range(len(transforred_data)): # along each sample!!!
#    print(LA.norm(transforred_data[i,:], 2))
#exit(0)
data_keys = list(emb_dict.keys())
for i in range(len(data_keys)):
    emb_dict[data_keys[i]] = transforred_data[i]
t_emb_norm_1 = time.time()
print(f'@t{t}, emb Normalizer L2  time cost: {(t_emb_norm_1-t_emb_norm_0):.2f}s')
"""
"""
#6 L1 all in one
t_emb_norm_0 = time.time()
from sklearn.preprocessing import Normalizer
data = list(emb_dict.values())
transforred_data = Normalizer(norm='l1').fit_transform(data)
#from numpy import linalg as LA
#for i in range(len(transforred_data)): # along each sample!!!
#    print(LA.norm(transforred_data[i,:], 1))
#exit(0)
data_keys = list(emb_dict.keys())
for i in range(len(data_keys)):
    emb_dict[data_keys[i]] = transforred_data[i]
t_emb_norm_1 = time.time()
print(f'@t{t}, emb Normalizer L1  time cost: {(t_emb_norm_1-t_emb_norm_0):.2f}s')
"""
"""
#7 L2 one model by one model
t_emb_norm_0 = time.time()
from sklearn.preprocessing import Normalizer
for key in emb_dict.keys():  # vectorized version... todo...
    v = []
    ind = 0
    for i in range(self.num_base_models):
        pre_ind = ind
        ind += self.emb_dim[i]
        v.extend( Normalizer(norm='l2').fit_transform([emb_dict[key][pre_ind:ind]]).tolist()[0] )
    #print(emb_dict[key])
    emb_dict[key] = v
    #print(emb_dict[key])
    #exit(0)
t_emb_norm_1 = time.time()
print(f'@t{t}, emb Normalizer L2 before concat time cost: {(t_emb_norm_1-t_emb_norm_0):.2f}s')
#exit(0)
"""
"""
#8 L1 one model by one model
t_emb_norm_0 = time.time()
from sklearn.preprocessing import Normalizer
for key in emb_dict.keys():  # vectorized version... todo...
    v = []
    ind = 0
    for i in range(self.num_base_models):
        pre_ind = ind
        ind += self.emb_dim[i]
        v.extend( Normalizer(norm='l1').fit_transform([emb_dict[key][pre_ind:ind]]).tolist()[0] )
    #print(emb_dict[key])
    emb_dict[key] = v
    #print(emb_dict[key])
    #exit(0)
t_emb_norm_1 = time.time()
print(f'@t{t}, emb Normalizer L1 before concat time cost: {(t_emb_norm_1-t_emb_norm_0):.2f}s')
#exit(0)
"""