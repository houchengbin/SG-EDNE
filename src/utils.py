"""
Commonly used utils:
evaluation metrics e.g. similarity, AUC, AP@k, MAP@k
graph related operation
testing data generator
file I/O
visualization
etc...
"""

import time
import numpy as np
from scipy import sparse
import pickle
import networkx as nx


# -----------------------------------------------------------------------------
# --------------------------------- metrics -----------------------------------
# -----------------------------------------------------------------------------
def cosine_similarity(a, b):
    from numpy import dot
    from numpy.linalg import norm
    ''' cosine similarity; can be used as score function; vector by vector; 
        If consider similarity for all pairs,
        pairwise_similarity() implementation may be more efficient
    '''
    a = np.reshape(a,-1)
    b = np.reshape(b,-1)
    if norm(a)*norm(b) == 0:
        return 0.0
    else:
        return dot(a, b)/(norm(a)*norm(b))

def pairwise_similarity(mat, type='cosine'):
    ''' pairwise similarity; can be used as score function;
        vectorized computation 
    '''
    if type == 'cosine':  # support sparse and dense mat
        from sklearn.metrics.pairwise import cosine_similarity
        result = cosine_similarity(mat, dense_output=True) # use sparse output to save ROM
    elif type == 'jaccard':
        from sklearn.metrics import jaccard_similarity_score
        from sklearn.metrics.pairwise import pairwise_distances
        # n_jobs=-1 means using all CPU for parallel computing
        result = pairwise_distances(mat.todense(), metric=jaccard_similarity_score, n_jobs=-1)
    elif type == 'euclidean':
        from sklearn.metrics.pairwise import euclidean_distances
        # note: similarity = - distance
        result = euclidean_distances(mat)
        result = -result
    elif type == 'manhattan':
        from sklearn.metrics.pairwise import manhattan_distances
        # note: similarity = - distance
        result = manhattan_distances(mat)
        result = -result
    else:
        print('Please choose from: cosine, jaccard, euclidean or manhattan')
        return 'Not found!'
    return result

def auc_score(y_true, y_score):
    ''' use sklearn roc_auc_score API
        y_true & y_score; array-like, shape = [n_samples]
    '''
    from sklearn.metrics import roc_auc_score
    roc = roc_auc_score(y_true=y_true, y_score=y_score)
    # it is OK in static case to reverse the prediction, but seems strange in dynamic case...
    #if roc < 0.5:
    #    roc = 1.0 - roc  # since binary clf, just predict the opposite if < 0.5
    return roc

def ranking_precision_score(y_true, y_score, k=10):
    ''' Precision at rank K or P@K
        y_true & y_score; array-like, shape = [n_samples]
        based on https://gist.github.com/mblondel/7337391
        but slightly changed based on https://ieeexplore.ieee.org/document/8329541
        e.g.
        y_true = [1, 0, 1, 0, 1, 0, 0, 0, 1]
        y_score = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        ranking_precision_score(y_true, y_score, k=1) = 1.0
        ranking_precision_score(y_true, y_score, k=2) = 0.5
        ranking_precision_score(y_true, y_score, k=3) ~ 0.667
        ranking_precision_score(y_true, y_score, k=4) = 0.5
        ranking_precision_score(y_true, y_score, k=5) = 0.6
        ranking_precision_score(y_true, y_score, k=6) = 0.5
        ranking_precision_score(y_true, y_score, k=7) ~ 0.429
        ranking_precision_score(y_true, y_score, k=8) = 0.375
        ranking_precision_score(y_true, y_score, k=9) ~ 0.444
    '''
    # unique_y = np.unique(y_true)
    # if len(unique_y) > 2:
    #    raise ValueError("Only supported for two relevance levels.")
    # pos_label = unique_y[1] # 1 as true   # zero degree -> index 1 is out of bounds 
    pos_label = 1  # a solution to fix zero degree node case
    order = np.argsort(y_score)[::-1] # return index based on ranked scores from high to low
    y_pred_true = np.take(y_true, order[:k]) # predict to be true @k based on the sorted y_score and its index
    n_relevant = np.sum(y_pred_true == pos_label) # predict to be true @k but how many of them are truly correct
    # Divide by min(n_pos, k) such that the best achievable score is always 1.0 (note: if k>n_pos, we use fixed n_pos; otherwise use given k)
    # n_pos = np.sum(y_true == pos_label)
    # return float(n_relevant) / min(n_pos, k)
    # instead we follow https://ieeexplore.ieee.org/document/8329541
    if k == 0:
        return 0
    return float(n_relevant) / k

def average_precision_score(y_true, y_score, k=10):
    ''' Average Precision at rank K @ AP@k
        y_true & y_score; array-like, shape = [n_samples]
        based on https://gist.github.com/mblondel/7337391
        but slightly changed based on https://ieeexplore.ieee.org/document/8329541
        e.g.
        y_true = [1, 0, 1, 0, 1, 0, 0, 0, 1]
        y_score = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        average_precision_score(y_true, y_score, k=1) = 1.0
        average_precision_score(y_true, y_score, k=2) = 1.0
        average_precision_score(y_true, y_score, k=3) ~ 0.833
        average_precision_score(y_true, y_score, k=4) ~ 0.833
        average_precision_score(y_true, y_score, k=5) ~ 0.756
        average_precision_score(y_true, y_score, k=6) ~ 0.756
        average_precision_score(y_true, y_score, k=7) ~ 0.756
        average_precision_score(y_true, y_score, k=8) ~ 0.756
        average_precision_score(y_true, y_score, k=9) ~ 0.678
    '''
    # unique_y = np.unique(y_true)
    # if len(unique_y) > 2:
    #    raise ValueError("Only supported for two relevance levels.")
    # pos_label = unique_y[1] # 1 as true  # zero degree -> index 1 is out of bounds
    pos_label = 1  # a solution to fix zero degree node case
    # n_pos = np.sum(y_true == pos_label)
    # order = np.argsort(y_score)[::-1][:min(n_pos, k)] # again, we did not follow https://gist.github.com/mblondel/7337391 
    order = np.argsort(y_score)[::-1][:k] # return 'top-k' index based on ranked scores from high to low
    y_pred_true = np.asarray(y_true)[order] # predict to be true @k based on the sorted y_score and its index
    score = 0
    for i in range(len(y_pred_true)): # len(y_pred_true)=k
        # only consider P@K=i if y_pred_true[i] = pas_label
        if y_pred_true[i] == pos_label:
            relevant = 0
            # calculate P@i
            for j in range(i + 1):
                if y_pred_true[j] == pos_label: # relevant!
                    relevant += 1.0
            relevant /= (i + 1.0)  # note: i+1.0 since i start from 0
            score += relevant
    # here we did not follow https://gist.github.com/mblondel/7337391
    #if n_pos == 0:
    #    return 0
    #return score / n_pos
    # instead we follow https://ieeexplore.ieee.org/document/8329541
    n_relevant = np.sum(y_pred_true == pos_label) # n_relevant = num of true positive = num of times of "score += relevant" executes!
    if n_relevant == 0:
        return 0
    return score / float(n_relevant)


def pk_and_apk_score(y_true, y_score, k_list):
    ''' P@k and AP@k at same time
        k here should be a list, e.g., k_list=[10] or k_list=[10, 100, 1000, ...]
        TODO: can be futher speeded up by vectorized implementation!
        e.g.
        y_true = [1, 0, 1, 0, 1, 0, 0, 0, 1]
        y_score = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        k_list = [1,2,3,4,5,6,7,8,9]
        pk_and_apk_score(y_true, y_score, k_list) -> 
        [1.0, 0.5, 0.6666666666666666, 0.5, 0.6, 0.5, 0.42857142857142855, 0.375, 0.4444444444444444], 
        [1.0, 1.0, 0.8333333333333333, 0.8333333333333333, 0.7555555555555555, 0.7555555555555555, 0.7555555555555555, 0.7555555555555555, 0.6777777777777778]
    '''
    pos_label = 1  
    order = np.argsort(y_score)[::-1] # sort once for all; to speedup
    pk_list = []
    apk_list = []
    for k in k_list:
        y_pred_true = np.take(y_true, order[:k])
        n_relevant = np.sum(y_pred_true == pos_label)
        if k == 0:
            pk_list.append(0)
        else: 
            pk_list.append(float(n_relevant)/k) # P@k
    
        score = 0
        for i in range(len(y_pred_true)):
            if y_pred_true[i] == pos_label:
                relevant = 0
                for j in range(i + 1): # calculate P@i
                    if y_pred_true[j] == pos_label:
                        relevant += 1.0
                relevant /= (i + 1.0) 
                score += relevant
        if n_relevant == 0:
            apk_list.append(0)
        else:
            apk_list.append(score/float(n_relevant)) # AP@k
    return pk_list, apk_list


# ----------------------------------------------------------------------------------
# ------------------------------- graph related operation --------------------------
# ----------------------------------------------------------------------------------
def edge_s1_minus_s0(s1, s0, is_directed=False):
    ''' s1 and s0: edge/node-pairs set
    '''
    if not is_directed:
        s1_reordered = set( (a,b) if a<b else (b,a) for a,b in s1 )
        s0_reordered = set( (a,b) if a<b else (b,a) for a,b in s0 )
        return s1_reordered-s0_reordered
    else:
        print('currently not support directed case')

def unique_nodes_from_edge_set(edge_set):
    ''' take out unique nodes from edge set
    '''
    unique_nodes = []
    for a, b in edge_set:
        if a not in unique_nodes:
            unique_nodes.append(a)
        if b not in unique_nodes:
            unique_nodes.append(b)
    return unique_nodes

def row_as_probdist(mat, dense_output=False, preserve_zeros=False):
    '''Make each row of matrix sums up to 1.0, i.e., a probability distribution.
    Support both dense and sparse matrix.

    Attributes
    ----------
    mat : scipy sparse matrix or dense matrix or numpy array
        The matrix to be normalized
    dense_output : bool
        whether forced dense output
    perserve_zeros : bool
        If False, for row with all entries 0, we normalize it to a vector with all entries 1/n.
        Leave 0 otherwise
    Returns
    -------
    dense or sparse matrix:
        return dense matrix if input is dense matrix or numpy array
        return sparse matrix for sparse matrix input
        (note: np.array & np.matrix are diff; and may cause some dim issues...)
    '''
    row_sum = np.array(mat.sum(axis=1)).ravel()  # type: np.array
    zero_rows = row_sum == 0
    row_sum[zero_rows] = 1
    diag = sparse.dia_matrix((1 / row_sum, 0), (mat.shape[0], mat.shape[0]))
    mat = diag.dot(mat)
    if not preserve_zeros:
        mat += sparse.csr_matrix(zero_rows.astype(int)).T.dot(sparse.csr_matrix(np.repeat(1 / mat.shape[1], mat.shape[1])))

    if dense_output and sparse.issparse(mat):
        return mat.todense()
    return mat



# ----------------------------------------------------------------------------
# --------------------------------- files I/O --------------------------------
# ----------------------------------------------------------------------------
def load_any_obj_pkl(path):
    ''' load any object from pickle file
    '''
    with open(path, 'rb') as f:
        any_obj = pickle.load(f)
    return any_obj

def save_any_obj_pkl(obj, path):
    ''' save any object to pickle file
    '''
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def save_emb(emb_dict, path):
    ''' save embeddings to a txt file
        nodeID emb_dim1 emb_dim2 ...
    '''
    node_num = len(emb_dict.keys())
    with open(path, 'w') as f:
        for node, vec in emb_dict.items():
            f.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))

def load_emb(path):
    ''' load embeddings to a txt file
        nodeID emb_dim1 emb_dim2 ...
    '''
    emb_dict = {}
    with open(path, 'r') as f:
        for l in f.readlines():
            vec = l.split()
            emb_dict[vec[0]] = np.array([float(x) for x in vec[1:]])
    return emb_dict

def load_edge_label(filename):
    ''' load edge label from a txt file
        nodeID1 nodeID2 edge_label(s)
    '''
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        line = fin.readline()
        if line == '':
            break
        vec = line.strip().split(' ')
        X.append(vec[:2])
        Y.append(vec[2])
    fin.close()
    return X, Y

def load_node_label(filename):
    ''' load node label from a txt file
        nodeID node_label(s)
    '''
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        line = fin.readline()
        if line == '':
            break
        vec = line.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y


"""
# -----------------------------------------------------------------------
# ------------------------------- others --------------------------------
# -----------------------------------------------------------------------
def dim_reduction(mat, dim=128, method='pca'):
    ''' dimensionality reduction: PCA, SVD, etc...
        dim = # of columns
    '''
    print('START dimensionality reduction using ' + method + ' ......')
    t1 = time.time()
    if method == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=dim, svd_solver='auto', random_state=None)
        mat_reduced = pca.fit_transform(mat)  # sklearn pca auto remove mean, no need to preprocess
    elif method == 'svd':
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=dim, n_iter=5, random_state=None)
        mat_reduced = svd.fit_transform(mat)
    else:  # to do... more methods... e.g. random projection, ica, t-sne...
        print('dimensionality reduction method not found......')
    t2 = time.time()
    print('END dimensionality reduction: {:.2f}s'.format(t2-t1))
    return mat_reduced

# ------------------------------------------------------------------------
# --------------------------data generator -----------------------------
# ------------------------------------------------------------------------
def gen_test_edge_wrt_changes(graph_t0, graph_t1):
    ''' input: two networkx graphs
        generate **changed** testing edges for link prediction task
        currently, we only consider pos_neg_ratio = 1.0
        return: pos_edges_with_label [(node1, node2, 1), (), ...]
                neg_edges_with_label [(node3, node4, 0), (), ...]
    '''
    G0 = graph_t0.copy() 
    G1 = graph_t1.copy() # use copy to avoid problem caused by G1.remove_node(node)
    edge_add = edge_s1_minus_s0(s1=set(G1.edges()), s0=set(G0.edges()))
    edge_del = edge_s1_minus_s0(s1=set(G0.edges()), s0=set(G1.edges()))
    unseen_nodes = set(G1.nodes()) - set(G0.nodes())
    for node in unseen_nodes: # to avoid unseen nodes while testing
        G1.remove_node(node)
    
    edge_add_unseen_node = [] # to avoid unseen nodes while testing
    #print('len(edge_add)', len(edge_add))
    for node in unseen_nodes: 
        for edge in edge_add:
            if node in edge:
                edge_add_unseen_node.append(edge)
    edge_add = edge_add - set(edge_add_unseen_node)
    #print('len(edge_add)', len(edge_add))
    pos_edges_with_label = [list(item+(1,)) for item in edge_add]
    neg_edges_with_label = [list(item+(0,)) for item in edge_del]
    if len(edge_add) > len(edge_del):
        num = len(edge_add) - len(edge_del)
        i = 0
        for non_edge in nx.non_edges(G1):
            if non_edge not in edge_del:
                neg_edges_with_label.append(list(non_edge+(0,)))
                i += 1
            if i >= num:
                break
    elif len(edge_add) < len(edge_del):
        num = len(edge_del) - len(edge_add)
        i = 0
        for edge in nx.edges(G1):
            if edge not in edge_add:
                pos_edges_with_label.append(list(edge+(1,)))
                i += 1
            if i >= num:
                break
    else: # len(edge_add) == len(edge_del)
        pass
    return pos_edges_with_label, neg_edges_with_label

def gen_test_edge_wrt_remove(graph, edges_removed, balance_ratio=1.0):
    ''' given a networkx graph and edges_removed;
        generate non_edges not in [both graph and edges_removed];
        return all_test_samples including [edges_removed (pos samples), non_edges (neg samples)];
        return format X=[[1,2],[2,4],...] Y=[1,0,...] where Y tells where corresponding element has a edge
    '''
    g = graph
    num_edges_removed = len(edges_removed)
    num_non_edges = int(balance_ratio * num_edges_removed)
    num = 0
    non_edges = []
    exist_edges = list(g.G.edges())+list(edges_removed)
    while num < num_non_edges:
        non_edge = list(np.random.choice(g.look_back_list, size=2, replace=False))
        if non_edge not in exist_edges:
            num += 1
            non_edges.append(non_edge)

    test_node_pairs = edges_removed + non_edges
    test_edge_labels = list(np.ones(num_edges_removed)) + list(np.zeros(num_non_edges))
    return test_node_pairs, test_edge_labels


def ranking_recall_score(y_true, y_score, k=10):
    ''' Recall at rank K or R@K
        y_true & y_score; array-like, shape = [n_samples]
        e.g.
        y_true = [1, 0, 1, 0, 1, 0, 0, 0, 1]
        y_score = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        ranking_recall_score(y_true, y_score, k=1) = 0.25
        ranking_recall_score(y_true, y_score, k=2) = 0.25
        ranking_recall_score(y_true, y_score, k=3) = 0.5
        ranking_recall_score(y_true, y_score, k=4) = 0.5
        ranking_recall_score(y_true, y_score, k=5) = 0.75
        ranking_recall_score(y_true, y_score, k=6) = 0.75
        ranking_recall_score(y_true, y_score, k=7) = 0.75
        ranking_recall_score(y_true, y_score, k=8) = 0.75
        ranking_recall_score(y_true, y_score, k=9) = 1.0
    '''
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    pos_label = 1  # a solution to fix zero degree node case
    n_pos = np.sum(y_true == pos_label)
    order = np.argsort(y_score)[::-1] # return index based on ranked scores from high to low
    y_pred_true = np.take(y_true, order[:k]) # predict to be true @k based on the sorted y_score and its index
    n_relevant = np.sum(y_pred_true == pos_label) # predict to be true @k but how many of them are truly correct
    if n_pos == 0:
        return 0
    return float(n_relevant) / n_pos

def average_recall_score(y_true, y_score, k=10):
    ''' Average Recall at rank K @ AR@k
        y_true & y_score; array-like, shape = [n_samples]
        e.g.
        y_true = [1, 0, 1, 0, 1, 0, 0, 0, 1]
        y_score = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        average_recall_score(y_true, y_score, k=1) = 0.25
        average_recall_score(y_true, y_score, k=2) = 0.25
        average_recall_score(y_true, y_score, k=3) = 0.375
        average_recall_score(y_true, y_score, k=4) = 0.375
        average_recall_score(y_true, y_score, k=5) = 0.5
        average_recall_score(y_true, y_score, k=6) = 0.5
        average_recall_score(y_true, y_score, k=7) = 0.5
        average_recall_score(y_true, y_score, k=8) = 0.5
        average_recall_score(y_true, y_score, k=9) = 0.625
    '''
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    pos_label = 1  # a solution to fix zero degree node case
    n_pos = np.sum(y_true == pos_label)
    order = np.argsort(y_score)[::-1][:k] # return 'top-k' index based on ranked scores from high to low
    y_pred_true = np.asarray(y_true)[order] # predict to be true @k based on the sorted y_score and its index
    score = 0
    for i in range(len(y_pred_true)): # len(y_pred_true)=k
        # only consider R@K=i if y_pred_true[i] = pas_label
        if y_pred_true[i] == pos_label:
            relevant = 0
            # calculate R@i
            for j in range(i + 1):
                if y_pred_true[j] == pos_label:  # predict to be true but whether it is correct
                    relevant += 1.0
            relevant /= n_pos
            score += relevant
    n_relevant = np.sum(y_pred_true == pos_label) # n_relevant = num of true positive = num of times of "score += relevant" executes!
    if n_relevant == 0:
        return 0
    return score / float(n_relevant)
"""