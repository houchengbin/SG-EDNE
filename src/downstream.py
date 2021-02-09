"""
downstream tasks: Link Prediction, Graph Reconstraction, Node Classification
each task is a Python Class
"""

import random
import numpy as np
import networkx as nx

# ----------------------------------------------------------------------------------
# ------------------ link prediction task based on AUC score -----------------------
# ----------------------------------------------------------------------------------
from utils import cosine_similarity, auc_score, edge_s1_minus_s0
from sklearn.linear_model import LogisticRegression
class lpClassifier(object):
    def __init__(self, emb_dict):
        self.embeddings = emb_dict

    # clf here is simply a similarity/distance metric <-- cosine_similarity(start_node_emb, end_node_emb)
    def evaluate_auc(self, X_test, Y_test):
        test_size = len(X_test)
        Y_true = [int(i) for i in Y_test]
        Y_score = []
        for i in range(test_size):
            start_node_emb = np.array(self.embeddings[X_test[i][0]]).reshape(-1, 1)
            end_node_emb = np.array(self.embeddings[X_test[i][1]]).reshape(-1, 1)
            score = cosine_similarity(start_node_emb, end_node_emb) # ranging from [-1, +1]
            Y_score.append(score)
        if len(Y_true) == 0:
            print('------- NOTE: two graphs do not have any change -> no testing data -> set result to 1......')
            auc = 1.0
        else:
            auc = auc_score(y_true=Y_true, y_score=Y_score)
        print("cos sim; auc=", "{:.9f}".format(auc))

    # clf here is a binary logistic regression <-- edge_feat via weighted-L1 as used in papers Node2Vec and DynamicTriad
    def lr_clf_init1(self, graph_t0):
        G0 = graph_t0.copy()
        pos_edges_with_label = [list(item+(1,)) for item in nx.edges(G0)]
        neg_edges_with_label = []
        num = len(pos_edges_with_label)
        i = 0
        for non_edge in nx.non_edges(G0):
            neg_edges_with_label.append(list(non_edge+(0,)))
            i += 1
            if i >= num:
                break
        all_edges_with_label = pos_edges_with_label + neg_edges_with_label
        random.seed(2021)
        random.shuffle(all_edges_with_label)
        all_test_edge = [e[:2] for e in all_edges_with_label]
        all_test_label = [e[2] for e in all_edges_with_label]
        test_size = len(all_test_edge)
        all_edge_feat = []
        for i in range(test_size):
            start_node_emb = np.array(self.embeddings[all_test_edge[i][0]])
            end_node_emb = np.array(self.embeddings[all_test_edge[i][1]])
            edge_feat = np.abs(start_node_emb-end_node_emb) # weighted-L1
            all_edge_feat.append(edge_feat)                 # weighted-L1 
        # print(np.shape(all_edge_feat))
        lr_clf_init = LogisticRegression(random_state=2021, penalty='l2', max_iter=1000).fit(all_edge_feat, all_test_label)
        return lr_clf_init

    def update_LR_auc1(self, X_test, Y_test, LR_prev=None):
        test_size = len(X_test)
        all_edge_feat = []
        for i in range(test_size):
            start_node_emb = np.array(self.embeddings[X_test[i][0]])
            end_node_emb = np.array(self.embeddings[X_test[i][1]])
            edge_feat = np.abs(start_node_emb-end_node_emb) # weighted-L1
            all_edge_feat.append(edge_feat)                 # weighted-L1
        lr_clf = LR_prev
        if len(Y_test) == 0:
            print('------- NOTE: two graphs do not have any change -> no testing data -> set result to 1......')
            auc = 1.0
        else:
            Y_score = lr_clf.predict_proba(all_edge_feat)[:,1]  # predict; the second col gives prob of positive edge
            auc = auc_score(y_true=Y_test, y_score=Y_score)
            lr_clf.fit(all_edge_feat, Y_test)                   # incremental update model parameters after predict
        print("weighted-L1; auc=", "{:.9f}".format(auc))
        return lr_clf

    # clf here is a binary logistic regression <-- edge_feat via weighted-L2 as used in papers Node2Vec
    def lr_clf_init2(self, graph_t0):
        G0 = graph_t0.copy()
        pos_edges_with_label = [list(item+(1,)) for item in nx.edges(G0)]
        neg_edges_with_label = []
        num = len(pos_edges_with_label)
        i = 0
        for non_edge in nx.non_edges(G0):
            neg_edges_with_label.append(list(non_edge+(0,)))
            i += 1
            if i >= num:
                break
        all_edges_with_label = pos_edges_with_label + neg_edges_with_label
        random.seed(2021)
        random.shuffle(all_edges_with_label)
        all_test_edge = [e[:2] for e in all_edges_with_label]
        all_test_label = [e[2] for e in all_edges_with_label]
        test_size = len(all_test_edge)
        all_edge_feat = []
        for i in range(test_size):
            start_node_emb = np.array(self.embeddings[all_test_edge[i][0]])
            end_node_emb = np.array(self.embeddings[all_test_edge[i][1]])
            edge_feat = np.abs(start_node_emb-end_node_emb) # weighted-L1
            edge_feat = edge_feat * edge_feat               # then weighted-L2
            all_edge_feat.append(edge_feat)
        # print(np.shape(all_edge_feat))
        lr_clf_init = LogisticRegression(random_state=2021, penalty='l2', max_iter=1000).fit(all_edge_feat, all_test_label)
        return lr_clf_init

    def update_LR_auc2(self, X_test, Y_test, LR_prev=None):
        test_size = len(X_test)
        all_edge_feat = []
        for i in range(test_size):
            start_node_emb = np.array(self.embeddings[X_test[i][0]])
            end_node_emb = np.array(self.embeddings[X_test[i][1]])
            edge_feat = np.abs(start_node_emb-end_node_emb) # weighted-L1
            edge_feat = edge_feat * edge_feat               # then weighted-L2
            all_edge_feat.append(edge_feat)  
        lr_clf = LR_prev
        if len(Y_test) == 0:
            print('------- NOTE: two graphs do not have any change -> no testing data -> set result to 1......')
            auc = 1.0
        else:
            Y_score = lr_clf.predict_proba(all_edge_feat)[:,1]  # predict; the second col gives prob of positive edge
            auc = auc_score(y_true=Y_test, y_score=Y_score)
            lr_clf.fit(all_edge_feat, Y_test)                   # incremental update model parameters after predict
        print("weighted-L2; auc=", "{:.9f}".format(auc))
        return lr_clf

    # clf here is a binary logistic regression <-- edge_feat via Hadamard as used in papers Node2Vec
    def lr_clf_init3(self, graph_t0):
        G0 = graph_t0.copy()
        pos_edges_with_label = [list(item+(1,)) for item in nx.edges(G0)]
        neg_edges_with_label = []
        num = len(pos_edges_with_label)
        i = 0
        for non_edge in nx.non_edges(G0):
            neg_edges_with_label.append(list(non_edge+(0,)))
            i += 1
            if i >= num:
                break
        all_edges_with_label = pos_edges_with_label + neg_edges_with_label
        random.seed(2021)
        random.shuffle(all_edges_with_label)
        all_test_edge = [e[:2] for e in all_edges_with_label]
        all_test_label = [e[2] for e in all_edges_with_label]
        test_size = len(all_test_edge)
        all_edge_feat = []
        for i in range(test_size):
            start_node_emb = np.array(self.embeddings[all_test_edge[i][0]])
            end_node_emb = np.array(self.embeddings[all_test_edge[i][1]])
            edge_feat = start_node_emb * end_node_emb # Hadamard
            all_edge_feat.append(edge_feat)
        # print(np.shape(all_edge_feat))
        lr_clf_init = LogisticRegression(random_state=2021, penalty='l2', max_iter=1000).fit(all_edge_feat, all_test_label)
        return lr_clf_init

    def update_LR_auc3(self, X_test, Y_test, LR_prev=None):
        test_size = len(X_test)
        all_edge_feat = []
        for i in range(test_size):
            start_node_emb = np.array(self.embeddings[X_test[i][0]])
            end_node_emb = np.array(self.embeddings[X_test[i][1]])
            edge_feat = start_node_emb * end_node_emb # Hadamard
            all_edge_feat.append(edge_feat)  
        lr_clf = LR_prev
        if len(Y_test) == 0:
            print('------- NOTE: two graphs do not have any change -> no testing data -> set result to 1......')
            auc = 1.0
        else:
            Y_score = lr_clf.predict_proba(all_edge_feat)[:,1]  # predict; the second col gives prob of positive edge
            auc = auc_score(y_true=Y_test, y_score=Y_score)
            lr_clf.fit(all_edge_feat, Y_test)                   # incremental update model parameters after predict
        print("Hadamard; auc=", "{:.9f}".format(auc))
        return lr_clf

    # clf here is a binary logistic regression <-- edge_feat via Average as used in papers Node2Vec
    def lr_clf_init4(self, graph_t0):
        G0 = graph_t0.copy()
        pos_edges_with_label = [list(item+(1,)) for item in nx.edges(G0)]
        neg_edges_with_label = []
        num = len(pos_edges_with_label)
        i = 0
        for non_edge in nx.non_edges(G0):
            neg_edges_with_label.append(list(non_edge+(0,)))
            i += 1
            if i >= num:
                break
        all_edges_with_label = pos_edges_with_label + neg_edges_with_label
        random.seed(2021)
        random.shuffle(all_edges_with_label)
        all_test_edge = [e[:2] for e in all_edges_with_label]
        all_test_label = [e[2] for e in all_edges_with_label]
        test_size = len(all_test_edge)
        all_edge_feat = []
        for i in range(test_size):
            start_node_emb = np.array(self.embeddings[all_test_edge[i][0]])
            end_node_emb = np.array(self.embeddings[all_test_edge[i][1]])
            edge_feat = (start_node_emb+end_node_emb)/2 # Average
            all_edge_feat.append(edge_feat)
        # print(np.shape(all_edge_feat))
        lr_clf_init = LogisticRegression(random_state=2021, penalty='l2', max_iter=1000).fit(all_edge_feat, all_test_label)
        return lr_clf_init

    def update_LR_auc4(self, X_test, Y_test, LR_prev=None):
        test_size = len(X_test)
        all_edge_feat = []
        for i in range(test_size):
            start_node_emb = np.array(self.embeddings[X_test[i][0]])
            end_node_emb = np.array(self.embeddings[X_test[i][1]])
            edge_feat = (start_node_emb+end_node_emb)/2 # Average
            all_edge_feat.append(edge_feat)  
        lr_clf = LR_prev
        if len(Y_test) == 0:
            print('------- NOTE: two graphs do not have any change -> no testing data -> set result to 1......')
            auc = 1.0
        else:
            Y_score = lr_clf.predict_proba(all_edge_feat)[:,1]  # predict; the second col gives prob of positive edge
            auc = auc_score(y_true=Y_test, y_score=Y_score)
            lr_clf.fit(all_edge_feat, Y_test)                   # incremental update model parameters after predict
        print("Average; auc=", "{:.9f}".format(auc))
        return lr_clf

def gen_test_edge_wrt_changes(graph_t0, graph_t1, seed=None):
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
    
    neg_edges_with_label = [list(item+(0,)) for item in edge_del]
    pos_edges_with_label = [list(item+(1,)) for item in edge_add]

    random.seed(seed)
    all_nodes = list(G0.nodes())

    if len(edge_add) > len(edge_del):
        num = len(edge_add) - len(edge_del)
        start_nodes = np.random.choice(all_nodes, num, replace=True)
        i = 0
        for start_node in start_nodes:
            try:
                non_nbrs = list(nx.non_neighbors(G0, start_node))
                non_nbr = random.sample(non_nbrs, 1).pop()
                non_edge = (start_node, non_nbr)
                if non_edge not in edge_del:
                    neg_edges_with_label.append(list(non_edge+(0,)))
                    i += 1
                if i >= num:
                    break
            except:
                print('Found a fully connected node: ', start_node, 'Ignore it...')
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
    print('---- len(pos_edges_with_label), len(neg_edges_with_label)', len(pos_edges_with_label), len(neg_edges_with_label))
    return pos_edges_with_label, neg_edges_with_label





# ------------------------------------------------------------------------------------------------------------------
#    ====== graph reconstruction and node recommendation task  ==== full version, but require large ROM ======
# ------------------------------------------------------------------------------------------------------------------
from utils import pairwise_similarity, ranking_precision_score, average_precision_score, pk_and_apk_score
class grClassifier(object): # graph reconstruction
    def __init__(self, emb_dict, rc_graph):
        self.graph = rc_graph
        self.adj_mat, self.score_mat = self.gen_test_data_wrt_graph_truth(graph=rc_graph, emb_dict=emb_dict)
    
    def gen_test_data_wrt_graph_truth(self, graph, emb_dict):
        ''' input: a networkx graph
            output: adj matrix and score matrix; note both matrices are symmetric
        '''
        adj_mat = nx.to_numpy_array(G=self.graph, nodelist=None) # ordered by G.nodes(); n-by-n
        adj_mat = np.where(adj_mat==0, False, True) # vectorized implementation weighted -> unweighted if necessary
        emb_mat = [emb_dict[node] for node in self.graph.nodes()]
        #emb_mat = []
        #for node in self.graph.nodes():
        #    emb_mat.append(emb_dict[node])
        score_mat = pairwise_similarity(emb_mat, type='cosine') # n-by-n corresponding to adj_mat
        n = len(score_mat)
        score_mat[range(n), range(n)] = 0.0  # set diagonal to 0 -> do not consider itself as the nearest neighbor (node without self loop)
        return np.array(adj_mat), np.array(score_mat)

    def evaluate_precision_k(self, top_k, node_list=None):
        ''' Precision at rank k
        '''
        pk_list = []
        if node_list==None: # eval all nodes
            size = self.adj_mat.shape[0] # num of rows -> num of nodes
            for i in range(size):
                pk_list.append(ranking_precision_score(self.adj_mat[i], self.score_mat[i], k=top_k)) # ranking_precision_score
        else: # only eval on node_list
            if len(node_list) == 0: # if there is no testing data, set auc to 1
                print('------- NOTE: no testing data -> set result to 1......')
                pk_list = 1.00
            else:
                node_idx = node_id2idx(self.graph, node_list)
                new_adj_mat = [self.adj_mat[i] for i in node_idx]
                new_score_mat = [self.score_mat[i] for i in node_idx]
                size = len(new_adj_mat)
                for i in range(size):
                    pk_list.append(ranking_precision_score(new_adj_mat[i], new_score_mat[i], k=top_k)) # ranking_precision_score only on node_list
        print("GR ranking_precision_score=", "{:.9f}".format(np.mean(pk_list))) # print mean of p@k

    def evaluate_average_precision_k(self, top_k, node_list=None):
        ''' Average precision at rank k
        '''
        apk_list = []
        if node_list==None: # eval all nodes
            size = self.adj_mat.shape[0] # num of rows -> num of nodes
            for i in range(size):
                apk_list.append(average_precision_score(self.adj_mat[i], self.score_mat[i], k=top_k)) # average_precision_score
        else: # only eval on node_list
            if len(node_list) == 0: # if there is no testing data, set auc to 1
                print('------- NOTE: no testing data -> set result to 1......')
                apk_list = 1.00
            else:
                node_idx = node_id2idx(self.graph, node_list)
                new_adj_mat = [self.adj_mat[i] for i in node_idx]
                new_score_mat = [self.score_mat[i] for i in node_idx]
                size = len(new_adj_mat)
                for i in range(size):
                    apk_list.append(average_precision_score(new_adj_mat[i], new_score_mat[i], k=top_k)) # average_precision_score only on node_list
        print("GR average_precision_score=", "{:.9f}".format(np.mean(apk_list))) # print mean of ap@k

    def evaluate_pk_and_apk(self, top_k_list, node_list=None):
        ''' Precision at rank k & Average precision at rank k
            k here should be a list, e.g., top_k_list=[10] or top_k_list=[10, 100, 1000, ...]
        '''
        all_pk_list = []
        all_apk_list = []
        if node_list==None: # eval all nodes
            size = self.adj_mat.shape[0] # num of rows -> num of nodes
            for i in range(size):
                pk_list, apk_list = pk_and_apk_score(self.adj_mat[i], self.score_mat[i], top_k_list)
                all_pk_list.append(pk_list)
                all_apk_list.append(apk_list)
        else: # only eval on node_list
            if len(node_list) == 0: # if there is no testing data, set auc to 1
                print('------- NOTE: no testing data -> set result to 1......')
                all_ones = [1.00]*len(top_k_list)
                all_pk_list.append(all_ones)
                all_apk_list.append(all_ones)
            else:
                node_idx = node_id2idx(self.graph, node_list)
                new_adj_mat = [self.adj_mat[i] for i in node_idx]
                new_score_mat = [self.score_mat[i] for i in node_idx]
                size = len(new_adj_mat)
                for i in range(size):
                    pk_list, apk_list = pk_and_apk_score(new_adj_mat[i], new_score_mat[i], top_k_list)
                    all_pk_list.append(pk_list)
                    all_apk_list.append(apk_list)
        all_pk_list = np.array(all_pk_list)
        all_pk = np.mean(all_pk_list, axis=0)  # mean over column for each top k
        all_apk_list = np.array(all_apk_list)
        all_apk = np.mean(all_apk_list, axis=0) # mean over column for each top k
        for i in range(len(top_k_list)):
            print("GR @ k= ", top_k_list[i])
            print("GR ranking_precision_score=", "{:.9f}".format(all_pk[i])) # print mean of P@k over each top_k_list
            print("GR average_precision_score=", "{:.9f}".format(all_apk[i])) # print mean of AP@k over each top_k_list


class nrClassifier(object): # node recommendation
    def __init__(self, emb_dict, rc_graph):
        self.graph = rc_graph
        self.adj_mat, self.score_mat = self.gen_test_data_wrt_graph_truth(graph=rc_graph, emb_dict=emb_dict)
    
    def gen_test_data_wrt_graph_truth(self, graph, emb_dict):
        ''' input: a networkx graph
            output: adj matrix and score matrix; note both matrices are symmetric
        '''
        adj_mat = nx.to_numpy_array(G=self.graph, nodelist=None) # ordered by G.nodes(); n-by-n
        adj_mat = np.where(adj_mat==0, False, True) # vectorized implementation weighted -> unweighted if necessary
        emb_mat = [emb_dict[node] for node in self.graph.nodes()]
        #emb_mat = []
        #for node in self.graph.nodes():
        #    emb_mat.append(emb_dict[node])
        score_mat = pairwise_similarity(emb_mat, type='cosine') # n-by-n corresponding to adj_mat
        n = len(score_mat)
        score_mat[range(n), range(n)] = 0.0  # set diagonal to 0 -> do not consider itself as the nearest neighbor (node without self loop)
        return np.array(adj_mat), np.array(score_mat)

    def evaluate_precision_k(self, top_k, node_list=None):
        ''' Precision at rank k
        '''
        pk_list = []
        if node_list==None: # eval all nodes
            size = self.adj_mat.shape[0] # num of rows -> num of nodes
            for i in range(size):
                pk_list.append(ranking_precision_score(self.adj_mat[i], self.score_mat[i], k=top_k)) # ranking_precision_score
        else: # only eval on node_list
            if len(node_list) == 0: # if there is no testing data, set auc to 1
                print('------- NOTE: no testing data -> set result to 1......')
                pk_list = 1.00
            else:
                node_idx = node_id2idx(self.graph, node_list)
                new_adj_mat = [self.adj_mat[i] for i in node_idx]
                new_score_mat = [self.score_mat[i] for i in node_idx]
                size = len(new_adj_mat)
                for i in range(size):
                    pk_list.append(ranking_precision_score(new_adj_mat[i], new_score_mat[i], k=top_k)) # ranking_precision_score only on node_list
        print("NR ranking_precision_score=", "{:.9f}".format(np.mean(pk_list))) # return mean of p@k

    def evaluate_average_precision_k(self, top_k, node_list=None):
        ''' Average precision at rank k
        '''
        apk_list = []
        if node_list==None: # eval all nodes
            size = self.adj_mat.shape[0] # num of rows -> num of nodes
            for i in range(size):
                apk_list.append(average_precision_score(self.adj_mat[i], self.score_mat[i], k=top_k)) # average_precision_score
        else: # only eval on node_list
            if len(node_list) == 0: # if there is no testing data, set auc to 1
                print('------- NOTE: no testing data -> set result to 1......')
                apk_list = 1.00
            else:
                node_idx = node_id2idx(self.graph, node_list)
                new_adj_mat = [self.adj_mat[i] for i in node_idx]
                new_score_mat = [self.score_mat[i] for i in node_idx]
                size = len(new_adj_mat)
                for i in range(size):
                    apk_list.append(average_precision_score(new_adj_mat[i], new_score_mat[i], k=top_k)) # average_precision_score only on node_list
        print("NR average_precision_score=", "{:.9f}".format(np.mean(apk_list))) # return mean of ap@k

    def evaluate_pk_and_apk(self, top_k_list, node_list=None):
        ''' Precision at rank k & Average precision at rank k
            k here should be a list, e.g., top_k_list=[10] or top_k_list=[10, 100, 1000, ...]
        '''
        all_pk_list = []
        all_apk_list = []
        if node_list==None: # eval all nodes
            size = self.adj_mat.shape[0] # num of rows -> num of nodes
            for i in range(size):
                pk_list, apk_list = pk_and_apk_score(self.adj_mat[i], self.score_mat[i], top_k_list)
                all_pk_list.append(pk_list)
                all_apk_list.append(apk_list)
        else: # only eval on node_list
            if len(node_list) == 0: # if there is no testing data, set auc to 1
                print('------- NOTE: no testing data -> set result to 1......')
                all_ones = [1.00]*len(top_k_list)
                all_pk_list.append(all_ones)
                all_apk_list.append(all_ones)
            else:
                node_idx = node_id2idx(self.graph, node_list)
                new_adj_mat = [self.adj_mat[i] for i in node_idx]
                new_score_mat = [self.score_mat[i] for i in node_idx]
                size = len(new_adj_mat)
                for i in range(size):
                    pk_list, apk_list = pk_and_apk_score(new_adj_mat[i], new_score_mat[i], top_k_list)
                    all_pk_list.append(pk_list)
                    all_apk_list.append(apk_list)
        all_pk_list = np.array(all_pk_list)
        all_pk = np.mean(all_pk_list, axis=0)  # mean over column for each top k
        all_apk_list = np.array(all_apk_list)
        all_apk = np.mean(all_apk_list, axis=0) # mean over column for each top k
        for i in range(len(top_k_list)):
            print("NR @ k= ", top_k_list[i])
            print("NR ranking_precision_score=", "{:.9f}".format(all_pk[i])) # print mean of P@k over each top_k_list
            print("NR average_precision_score=", "{:.9f}".format(all_apk[i])) # print mean of AP@k over each top_k_list


def node_id2idx(graph, node_id):
    G = graph
    all_nodes = list(G.nodes())
    node_idx = []
    for node in node_id:
        node_idx.append(all_nodes.index(node))
    return node_idx

def gen_test_node_wrt_changes(graph_t0, graph_t1):
    ''' generate the testing nodes that we are intereted
        here we take the affected nodes presented in both graphs
    '''
    from utils import unique_nodes_from_edge_set
    G0 = graph_t0.copy() 
    G1 = graph_t1.copy() # use copy to avoid problem caused by G1.remove_node(node)
    edge_add = edge_s1_minus_s0(s1=set(G1.edges()), s0=set(G0.edges()))
    edge_del = edge_s1_minus_s0(s1=set(G0.edges()), s0=set(G1.edges()))
    node_affected_by_edge_add = unique_nodes_from_edge_set(edge_add) # unique
    node_affected_by_edge_del = unique_nodes_from_edge_set(edge_del) # unique
    node_affected = list(set(node_affected_by_edge_add + node_affected_by_edge_del)) # unique
    node_add = [node for node in node_affected_by_edge_add if node not in G0.nodes()] # nodes not exist in G0
    node_del = [node for node in node_affected_by_edge_del if node not in G1.nodes()] # nodes not exist in G1

    not_intereted_node = node_add + node_del
    test_nodes = [node for node in node_affected if node not in not_intereted_node] # remove unseen nodes
    return test_nodes

def align_nodes(graph_t0, graph_t1):
    ''' remove newly added nodes from graph_t1, and add newly removed nodes to graph_t1
    '''
    from utils import unique_nodes_from_edge_set
    G0 = graph_t0.copy()
    G1 = graph_t1.copy()
    edge_add = edge_s1_minus_s0(s1=set(G1.edges()), s0=set(G0.edges()))  # one may directly use edge streams if available
    edge_del = edge_s1_minus_s0(s1=set(G0.edges()), s0=set(G1.edges()))
    node_affected_by_edge_add = unique_nodes_from_edge_set(edge_add)
    node_affected_by_edge_del = unique_nodes_from_edge_set(edge_del)
    node_affected = list(set(node_affected_by_edge_add + node_affected_by_edge_del))  # nodes being directly affected between G0 and G1
    node_add = [node for node in node_affected_by_edge_add if node not in G0.nodes()] # nodes not exist in G0
    node_del = [node for node in node_affected_by_edge_del if node not in G1.nodes()] # nodes not exist in G1
    # to align G1 with G0
    G1.remove_nodes_from(node_add)
    G1.add_nodes_from(node_del)
    return G1


"""
# ------------------------------------------------------------------------------------------------------------------
#   ====== graph reconstruction and node recommendation task  ==== batch version, memory saving but slow ======
# ------------------------------------------------------------------------------------------------------------------
from utils import pairwise_similarity, ranking_precision_score, average_precision_score
class grClassifier_batch(object): # graph reconstruction batch version
    def __init__(self, emb_dict, rc_graph):
        self.embeddings = emb_dict
        self.graph = rc_graph
        self.adj_mat = nx.to_numpy_array(G=self.graph, nodelist=self.graph.nodes()) # ordered by G.nodes(); n-by-n
        self.adj_mat = np.where(self.adj_mat==0, False, True) # vectorized implementation weighted -> unweighted if necessary; if still OOM, try sparse matrix
        self.emb_mat = [self.embeddings[node] for node in self.graph.nodes()] # ordered by G.nodes(); n-by-n
        self.norm_vec = np.sqrt(np.einsum('ij,ij->i', self.emb_mat, self.emb_mat)) # norm of each row of emb_mat  # https://stackoverflow.com/questions/7741878/how-to-apply-numpy-linalg-norm-to-each-row-of-a-matrix

    def evaluate_precision_k(self, top_k, node_list=None):
        ''' Precision at rank k
        '''
        pk_list = []
        num_nodes = len(self.graph.nodes())
        if node_list == None: # eval all nodes
            for i in range(num_nodes): # due to memory issue, we have to eval them one by one...
                score = [np.inner(self.emb_mat[i],self.emb_mat[j])/(self.norm_vec[i]*self.norm_vec[j]) for j in range(num_nodes)] # cos of i w.r.t each j
                pk_list.append(ranking_precision_score(y_true=self.adj_mat[i], y_score=score, k=top_k))
        else: # only eval on node_list
            if len(node_list) == 0: # if there is no testing data (dyn networks not changed), set auc to 1
                print('------- NOTE: two graphs do not have any change -> no testing data -> set result to 1......')
                pk_list = 1.00
            else:
                node_idx = node_id2idx(self.graph, node_list)
                for i in node_idx: # only eval on node_list
                    score = [np.inner(self.emb_mat[i],self.emb_mat[j])/(self.norm_vec[i]*self.norm_vec[j]) for j in range(num_nodes)] # cos of i w.r.t each j
                    pk_list.append(ranking_precision_score(y_true=self.adj_mat[i], y_score=score, k=top_k))
        print("GR ranking_precision_score=", "{:.9f}".format(np.mean(pk_list))) # return mean of p@k

    def evaluate_average_precision_k(self, top_k, node_list=None):
        ''' Average Precision at rank k
        '''
        apk_list = []
        num_nodes = len(self.graph.nodes())
        if node_list == None: # eval all nodes
            for i in range(num_nodes): # due to memory issue, we have to eval them one by one...
                score = [np.inner(self.emb_mat[i],self.emb_mat[j])/(self.norm_vec[i]*self.norm_vec[j]) for j in range(num_nodes)] # cos of i w.r.t each j
                apk_list.append(average_precision_score(y_true=self.adj_mat[i], y_score=score, k=top_k))
        else: # only eval on node_list
            if len(node_list) == 0: # if there is no testing data (dyn networks not changed), set auc to 1
                print('------- NOTE: two graphs do not have any change -> no testing data -> set result to 1......')
                apk_list = 1.00
            else:
                node_idx = node_id2idx(self.graph, node_list)
                for i in node_idx: # only eval on node_list
                    score = [np.inner(self.emb_mat[i],self.emb_mat[j])/(self.norm_vec[i]*self.norm_vec[j]) for j in range(num_nodes)] # cos of i w.r.t each j
                    apk_list.append(average_precision_score(y_true=self.adj_mat[i], y_score=score, k=top_k))
        print("GR average_precision_score=", "{:.9f}".format(np.mean(apk_list))) # return mean of ap@k


class nrClassifier_batch(object): # node recommendation batch version
    def __init__(self, emb_dict, rc_graph):
        self.embeddings = emb_dict
        self.graph = rc_graph
        self.adj_mat = nx.to_numpy_array(G=self.graph, nodelist=self.graph.nodes()) # ordered by G.nodes(); n-by-n
        self.adj_mat = np.where(self.adj_mat==0, False, True) # vectorized implementation weighted -> unweighted if necessary; if still OOM, try sparse matrix
        self.emb_mat = [self.embeddings[node] for node in self.graph.nodes()] # ordered by G.nodes(); n-by-n
        self.norm_vec = np.sqrt(np.einsum('ij,ij->i', self.emb_mat, self.emb_mat)) # norm of each row of emb_mat  # https://stackoverflow.com/questions/7741878/how-to-apply-numpy-linalg-norm-to-each-row-of-a-matrix

    def evaluate_precision_k(self, top_k, node_list=None):
        ''' Precision at rank k
        '''
        pk_list = []
        num_nodes = len(self.graph.nodes())
        if node_list == None: # eval all nodes
            for i in range(num_nodes): # due to memory issue, we have to eval them one by one...
                score = [np.inner(self.emb_mat[i],self.emb_mat[j])/(self.norm_vec[i]*self.norm_vec[j]) for j in range(num_nodes)] # cos of i w.r.t each j
                pk_list.append(ranking_precision_score(y_true=self.adj_mat[i], y_score=score, k=top_k))
        else: # only eval on node_list
            if len(node_list) == 0: # if there is no testing data (dyn networks not changed), set auc to 1
                print('------- NOTE: two graphs do not have any change -> no testing data -> set result to 1......')
                pk_list = 1.00
            else:
                node_idx = node_id2idx(self.graph, node_list)
                for i in node_idx: # only eval on node_list
                    score = [np.inner(self.emb_mat[i],self.emb_mat[j])/(self.norm_vec[i]*self.norm_vec[j]) for j in range(num_nodes)] # cos of i w.r.t each j
                    pk_list.append(ranking_precision_score(y_true=self.adj_mat[i], y_score=score, k=top_k))
        print("NR ranking_precision_score=", "{:.9f}".format(np.mean(pk_list))) # return mean of p@k

    def evaluate_average_precision_k(self, top_k, node_list=None):
        ''' Average Precision at rank k
        '''
        apk_list = []
        num_nodes = len(self.graph.nodes())
        if node_list == None: # eval all nodes
            for i in range(num_nodes): # due to memory issue, we have to eval them one by one...
                score = [np.inner(self.emb_mat[i],self.emb_mat[j])/(self.norm_vec[i]*self.norm_vec[j]) for j in range(num_nodes)] # cos of i w.r.t each j
                apk_list.append(average_precision_score(y_true=self.adj_mat[i], y_score=score, k=top_k))
        else: # only eval on node_list
            if len(node_list) == 0: # if there is no testing data (dyn networks not changed), set auc to 1
                print('------- NOTE: two graphs do not have any change -> no testing data -> set result to 1......')
                apk_list = 1.00
            else:
                node_idx = node_id2idx(self.graph, node_list)
                for i in node_idx: # only eval on node_list
                    score = [np.inner(self.emb_mat[i],self.emb_mat[j])/(self.norm_vec[i]*self.norm_vec[j]) for j in range(num_nodes)] # cos of i w.r.t each j
                    apk_list.append(average_precision_score(y_true=self.adj_mat[i], y_score=score, k=top_k))
        print("NR average_precision_score=", "{:.9f}".format(np.mean(apk_list))) # return mean of ap@k
"""


"""
# ----------------------------------------------------------------------------------
# ------------- node classification task based on F1 score -------------------------
# ----------------------------------------------------------------------------------
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
class ncClassifier(object):
    def __init__(self, emb_dict, clf):
        self.embeddings = emb_dict
        self.clf = TopKRanker(clf)  # here clf is LR
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def split_train_evaluate(self, X, Y, train_precent, seed=None):
        np.random.seed(seed=seed)
        state = np.random.get_state()
        training_size = int(train_precent * len(X))
        shuffle_indices = np.random.permutation(np.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]
        self.train(X_train, Y_train, Y)
        np.random.set_state(state)
        return self.evaluate(X_test, Y_test)

    def train(self, X, Y, Y_all):
        # to support multi-labels, fit means dict mapping {orig cat: binarized vec}
        self.binarizer.fit(Y_all)
        X_train = [self.embeddings[x] for x in X]
        # since we have use Y_all fitted, then we simply transform
        Y = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y)

    def predict(self, X, top_k_list):
        X_ = np.asarray([self.embeddings[x] for x in X])
        # see TopKRanker(OneVsRestClassifier)
        # the top k probs to be output...
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def evaluate(self, X, Y):
        # multi-labels, diff len of labels of each node
        top_k_list = [len(l) for l in Y]
        Y_ = self.predict(X, top_k_list)  # pred val of X_test i.e. Y_pred
        Y = self.binarizer.transform(Y)  # true val i.e. Y_test
        averages = ["micro", "macro", "samples", "weighted"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)
        print(results)
        return results

class TopKRanker(OneVsRestClassifier):  # orignal LR or SVM is for binary clf
    def predict(self, X, top_k_list):  # re-define predict func of OneVsRestClassifier
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[
                probs_.argsort()[-k:]].tolist()  # denote labels
            probs_[:] = 0  # reset probs_ to all 0
            probs_[labels] = 1  # reset probs_ to 1 if labels denoted...
            all_labels.append(probs_)
        return np.asarray(all_labels)
"""



"""
    # clf here is a binary logistic regression <-- np.concatenate((start_node_emb, end_node_emb))
    def lr_clf_init(self, graph_t0):
        G0 = graph_t0.copy()
        pos_edges_with_label = [list(item+(1,)) for item in nx.edges(G0)]
        neg_edges_with_label = []
        num = len(pos_edges_with_label)
        i = 0
        for non_edge in nx.non_edges(G0):
            neg_edges_with_label.append(list(non_edge+(0,)))
            i += 1
            if i >= num:
                break
        all_edges_with_label = pos_edges_with_label + neg_edges_with_label
        random.seed(2021)
        random.shuffle(all_edges_with_label)
        all_test_edge = [e[:2] for e in all_edges_with_label]
        all_test_label = [e[2] for e in all_edges_with_label]
        test_size = len(all_test_edge)
        all_edge_feat = []
        for i in range(test_size):
            start_node_emb = np.array(self.embeddings[all_test_edge[i][0]])
            end_node_emb = np.array(self.embeddings[all_test_edge[i][1]])
            all_edge_feat.append(np.concatenate((start_node_emb, end_node_emb)))
        # print(np.shape(all_edge_feat))
        lr_clf_init = LogisticRegression(random_state=2021, penalty='l2', max_iter=1000).fit(all_edge_feat, all_test_label)
        return lr_clf_init

    def update_LR_auc(self, X_test, Y_test, LR_prev=None):
        test_size = len(X_test)
        all_edge_feat = []
        for i in range(test_size):
            start_node_emb = np.array(self.embeddings[X_test[i][0]])
            end_node_emb = np.array(self.embeddings[X_test[i][1]])
            all_edge_feat.append(np.concatenate((start_node_emb, end_node_emb)))
            
        lr_clf = LR_prev
        if len(Y_test) == 0:
            print('------- NOTE: two graphs do not have any change -> no testing data -> set result to 1......')
            auc = 1.0
        else:
            Y_score = lr_clf.predict_proba(all_edge_feat)[:,1]  # predict; the second col gives prob of true
            auc = auc_score(y_true=Y_test, y_score=Y_score)
            lr_clf.fit(all_edge_feat, Y_test)  # update model parameters
        print("concat; auc=", "{:.9f}".format(auc))
        return lr_clf
"""