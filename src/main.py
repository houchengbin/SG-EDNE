"""
code for paper: Robust Dynamic Network Embedding via Ensembles

our approach: Skip-Gram based Ensembles Dynamic Network Embedding (SG-EDNE)
Step 1: prepare data
Step 2: learn node embeddings
Step 3: evaluate downstream tasks
"""

import time
import pickle
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import numpy as np
import networkx as nx
#import logging
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    # general settings
    parser.add_argument('--graph', default='data/DNC-Email_LCC.pkl',
                        help='graph/network')
    #parser.add_argument('--label', default='data/cora_node_label_dict.pkl',
    #                    help='node label')
    parser.add_argument('--emb-file', default='output/DNC-Email_LCC_SG-EDNE_emb128.pkl',
                        help='node embeddings; suggest: data_method_emb_dim.pkl')
    parser.add_argument('--emb-dim', default=128, type=int,
                        help='node embeddings dimensions')
    parser.add_argument('--task', default='all', choices=['gr', 'lp', 'nr', 'nc', 'all', 'save'],
                        help='choices of downstream tasks: gr, lp, nc, all, save')
    parser.add_argument('--seed', default=2021, type=int,
                        help='random seed to fix testing data')
    # ensemble settings
    parser.add_argument('--method', default='SG-EDNE', choices=['SG-EDNE', 'SG-EDNE-fixRW', 'SG-SNE'],
                        help='choices of Network Embedding methods')
    parser.add_argument('--num-base-models', default=3, type=int,
                        help='# of base models to ensemble')
    parser.add_argument('--restart', default=0, type=int,
                        help='0 False: random walk; 1 True: random walk with restart')
    parser.add_argument('--max-r-prob', default=0.1, type=float,
                        help='max-r-prob')
    parser.add_argument('--scaling_strategy', default=0, type=int,
                        help='scaling strategy 0: do nothing; 1: [0~1]; 2: [-1~1]; 3: (0 mean, 1 var)')
    parser.add_argument('--sampling_strategy', default=1, type=int,
                        help='sampling strategy 1: repeat samples; 2: random samples; 3: frac of random samples')
    parser.add_argument('--combining_strategy', default=1, type=int,
                        help='combining strategy 1: concat; 2: mean; 3: sum; 4: min; 5 max')
    # random walks settings
    parser.add_argument('--num-walks', default=10, type=int,
                        help='# of random walks of each node')
    parser.add_argument('--walk-length', default=80, type=int,
                        help='length of each random walk')
    # gensim word2vec settings
    parser.add_argument('--window', default=10, type=int,
                        help='window size of SG model')
    parser.add_argument('--negative', default=5, type=int,
                        help='negative samples of SG model')
    parser.add_argument('--workers', default=32, type=int,
                        help='# of parallel processes.')
    args = parser.parse_args()
    return args


def main(args):
    # -------- Step 1: prepare data --------
    print(f'Summary of all settings: {args}')
    print('\nStep 1: start loading data ...')
    t1 = time.time()
    with open(args.graph, 'rb') as f:
        G_dynamic = pickle.load(f)
    t2 = time.time()
    print(f'Step 1: end loading data; time cost: {(t2-t1):.2f}s')

    # -------- Step 2: upstream embedding task --------
    print('\nStep 2: start learning embeddings ...')
    print(f'The model being used: {args.method} \
            \nThe # of dynamic graphs: {len(G_dynamic)}; \
            \nThe # of nodes @t_init: {nx.number_of_nodes(G_dynamic[0])}, and @t_last {nx.number_of_nodes(G_dynamic[-1])} \
            \nThe # of edges @t_init: {nx.number_of_edges(G_dynamic[0])}, and @t_last {nx.number_of_edges(G_dynamic[-1])}')
    t1 = time.time()
    model = None
    if args.method == 'SG-EDNE': # Skip-Gram based Ensembles Dynamic Network Embedding
        import sg_edne
        model = sg_edne.SG_EDNE(G_dynamic=G_dynamic, num_walks=args.num_walks, walk_length=args.walk_length, emb_dim=args.emb_dim, 
                                window=args.window, negative=args.negative, workers=args.workers, seed=args.seed, 
                                num_base_models=args.num_base_models, restart=args.restart, max_r_prob=args.max_r_prob,
                                sampling_strategy=args.sampling_strategy, combining_strategy=args.combining_strategy, scaling_strategy=args.scaling_strategy) 
        model.traning()
    elif args.method == 'SG-EDNE-fixRW': # Skip-Gram based Ensembles Dynamic Network Embedding with fixed Random Walks
        import sg_edne_fixrw
        model = sg_edne_fixrw.SG_EDNE_fixRW(G_dynamic=G_dynamic, num_walks=args.num_walks, walk_length=args.walk_length, emb_dim=args.emb_dim, 
                                window=args.window, negative=args.negative, workers=args.workers, seed=args.seed, 
                                num_base_models=args.num_base_models,
                                sampling_strategy=args.sampling_strategy, combining_strategy=args.combining_strategy, scaling_strategy=args.scaling_strategy) 
        model.traning()
    elif args.method == 'SG-SNE': # Skip-Gram based Static Network Embedding at each time step
        import sg_sne
        model = sg_sne.SG_SNE(G_dynamic=G_dynamic, num_walks=args.num_walks, walk_length=args.walk_length, emb_dim=args.emb_dim, 
                                window=args.window, negative=args.negative, workers=args.workers, seed=args.seed, scaling_strategy=args.scaling_strategy) 
        model.traning()
    else:
        exit('Exit, method not found ...')
    t2 = time.time()
    print(f'Step 2: end learning embeddings; time cost: {(t2-t1):.2f}s')

    # -------- Step 3: downstream task --------
    print('\n\nStep 3: start evaluating embeddings ...')
    t1 = time.time()
    emb_dicts = model.emb_dicts
    del model  # to save memory
    if args.task == 'save':
        with open(args.emb_file, 'wb') as f:
            pickle.dump(emb_dicts, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Save node embeddings in file: {args.emb_file}')
        exit('emb saved, no downsateam task, exit... ')
    
    print(time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()))
    if args.task == 'lp' or args.task == 'all':
        from downstream import lpClassifier, gen_test_edge_wrt_changes
        # the size of LP testing data depends on the changes between two consecutive snapshots
        test_edges = []
        test_labels = []
        for t in range(len(G_dynamic)-1): # changed edges from t to t+1 as testing edges
            pos_edges_with_label, neg_edges_with_label = gen_test_edge_wrt_changes(G_dynamic[t], G_dynamic[t+1], seed=args.seed)
            test_edges.append( [e[:2] for e in pos_edges_with_label] + [e[:2] for e in neg_edges_with_label] )
            test_labels.append( [e[2] for e in pos_edges_with_label] + [e[2] for e in neg_edges_with_label] )
        # ====== Changed Link Prediction task (via cos sim) by AUC score ======
        print('--- Start changed link prediction task --> use current emb @t to predict **future** changed links @t+1: ')
        for t in range(len(G_dynamic)-1):
            print(f'Current time step @t: {t}')
            print(f'Changed Link Prediction task (via cos sim) by AUC score')
            ds_task = lpClassifier(emb_dict=emb_dicts[t]) # emb at t; did not use **future** changed edges
            ds_task.evaluate_auc(test_edges[t], test_labels[t]) # evalate prediction of changed edges from t to t+1
        # ====== Changed Link Prediction task (Weighted-L1 edge_feat --> LR clf) by AUC score ======
        print(f'--- start changed link prediction task 1 --> use current emb @t to predict **future** changed links @t+1: ')
        LR_prev = None
        for t in range(len(G_dynamic)-1):
            print(f'Current time step @t: {t}')
            print(f'Changed Link Prediction task (Weighted-L1 edge_feat --> LR clf) by AUC score')
            ds_task = lpClassifier(emb_dict=emb_dicts[t])
            if t == 0:
                LR_prev = ds_task.lr_clf_init1(G_dynamic[t])
            LR_prev = ds_task.update_LR_auc1(test_edges[t], test_labels[t], LR_prev=LR_prev) # incremental learning for Changed LP task; LogisticRegression(random_state=2021, penalty='l2', max_iter=1000)
        # ====== Changed Link Prediction task (Weighted-L2 edge_feat --> LR clf) by AUC score ======
        print(f'--- start changed link prediction task 2 --> use current emb @t to predict **future** changed links @t+1: ')
        LR_prev = None
        for t in range(len(G_dynamic)-1):
            print(f'Current time step @t: {t}')
            print(f'Changed Link Prediction task (Weighted-L2 edge_feat --> LR clf) by AUC score')
            ds_task = lpClassifier(emb_dict=emb_dicts[t])
            if t == 0:
                LR_prev = ds_task.lr_clf_init2(G_dynamic[t])
            LR_prev = ds_task.update_LR_auc2(test_edges[t], test_labels[t], LR_prev=LR_prev) # incremental learning for Changed LP task; LogisticRegression(random_state=2021, penalty='l2', max_iter=1000)
        # ====== Changed Link Prediction task (Hadamard edge_feat --> LR clf) by AUC score ======
        print(f'--- start changed link prediction task 3 --> use current emb @t to predict **future** changed links @t+1: ')
        LR_prev = None
        for t in range(len(G_dynamic)-1):
            print(f'Current time step @t: {t}')
            print(f'Changed Link Prediction task (Hadamard edge_feat --> LR clf) by AUC score')
            ds_task = lpClassifier(emb_dict=emb_dicts[t])
            if t == 0:
                LR_prev = ds_task.lr_clf_init3(G_dynamic[t])
            LR_prev = ds_task.update_LR_auc3(test_edges[t], test_labels[t], LR_prev=LR_prev) # incremental learning for Changed LP task; LogisticRegression(random_state=2021, penalty='l2', max_iter=1000)
        # ====== Changed Link Prediction task (Average edge_feat --> LR clf) by AUC score ======
        print(f'--- start changed link prediction task 4 --> use current emb @t to predict **future** changed links @t+1: ')
        LR_prev = None
        for t in range(len(G_dynamic)-1):
            print(f'Current time step @t: {t}')
            print(f'Changed Link Prediction task (Average edge_feat --> LR clf) by AUC score')
            ds_task = lpClassifier(emb_dict=emb_dicts[t])
            if t == 0:
                LR_prev = ds_task.lr_clf_init4(G_dynamic[t])
            LR_prev = ds_task.update_LR_auc4(test_edges[t], test_labels[t], LR_prev=LR_prev) # incremental learning for Changed LP task; LogisticRegression(random_state=2021, penalty='l2', max_iter=1000)
    
    print(time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()))
    if args.task == 'nr' or args.task == 'all':
        print(f'--- start changed node recommendation task --> use current emb @t to recommend nodes for **future** changed node in graph @t+1: ')
        from downstream import nrClassifier, gen_test_node_wrt_changes, align_nodes
        for t in range(len(G_dynamic)-1):
            print(f'Current time step @t: {t}')
            node_list = gen_test_node_wrt_changes(G_dynamic[t],G_dynamic[t+1]) # generate the testing nodes that affected by changes and presented in both graphs
            print('# of testing nodes that affected by changes and presented in both graphs: ', len(node_list))
            rc_next_graph_aligned = align_nodes(G_dynamic[t], G_dynamic[t+1])  # remove newly added nodes from G_dynamic[t+1], and add newly removed nodes to G_dynamic[t+1]
            ds_task = nrClassifier(emb_dict=emb_dicts[t], rc_graph=rc_next_graph_aligned)
            top_k_list = [5, 10, 50, 100]
            ds_task.evaluate_pk_and_apk(top_k_list, node_list)
            # If OOM, try grClassifier_batch (see dowmstream.py) which is slow but requires much smaller memory
    
    print(time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()))
    if args.task == 'gr' or args.task == 'all':
        print(f'--- start graph/link reconstraction task --> use current emb @t to reconstruct **current** graph @t: ')
        from downstream import grClassifier
        for t in range(len(G_dynamic)-1): # ignore the last one, so that the length is consistent with Changed LP
            print(f'Current time step @t: {t}')
            all_nodes = list(G_dynamic[t].nodes())
            if len(all_nodes) <= 10000:
                node_list = None  # testing all nodes
                print('# testing for all nodes in current graph')
            else:
                node_list = list(np.random.choice(all_nodes, 10000, replace=False))
                print('# current graph is too larger -> randomly sample 10000 testing nodes: ', len(node_list))
            ds_task = grClassifier(emb_dict=emb_dicts[t], rc_graph=G_dynamic[t])
            top_k_list = [5, 10, 50, 100]
            ds_task.evaluate_pk_and_apk(top_k_list, node_list)
            # If OOM, try grClassifier_batch (see dowmstream.py) which is slow but requires much smaller memory

    t2 = time.time()
    print(f'STEP3: end evaluating; time cost: {(t2-t1):.2f}s')
    print(time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()))


if __name__ == '__main__':
    print(f'------ START @ {time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())} ------')
    main(parse_args())
    print(f'------ END @ {time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())} ------')
    