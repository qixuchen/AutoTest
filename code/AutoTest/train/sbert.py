import os
import pickle
import numpy as np
from util import sbert_utils


DFT_COV_THRES = 0.0003
DFT_SAMPLE_SIZE = 50
DFT_RATIO_LIST = [0.8]
DFT_LB_LIST = [0.8 + 0.1 * i for i in range(8)]
DFT_UB_LIST = [1.1 + 0.025 * i for i in range(17)]
DFT_SBERT_EMBED_FNAME = os.path.join(os.getcwd(), 'sbert_embedding.pkl')
DFT_NPROC = 8

def rule_generate(train, params):
    coverage_thres = params['coverage_thres'] if 'coverage_thres' in params else DFT_COV_THRES
    sample_size = params['sample_size'] if 'sample_size' in params else DFT_SAMPLE_SIZE
    ratio_list = params['ratio_list'] if 'ratio_list' in params else DFT_RATIO_LIST
    lb_list = params['lb_list'] if 'lb_list' in params else DFT_LB_LIST
    ub_list = params['ub_list'] if 'ub_list' in params else DFT_UB_LIST
    sbert_dist_val_embeddings_fname = params['sbert_dist_val_embeddings_fname'] if 'sbert_dist_val_embeddings_fname' in params else DFT_SBERT_EMBED_FNAME
    n_proc = params['n_proc'] if 'n_proc' in params else DFT_NPROC
    
    if not os.path.exists(sbert_dist_val_embeddings_fname):
        print("SentenceBERT embedding file not found, computing ...")
        print(f'Results will be saved to {sbert_dist_val_embeddings_fname}.')
        sbert_dist_val_embeddings = sbert_utils.dist_val_embeddings_parallel(train, n_proc)
        with open(sbert_dist_val_embeddings_fname, 'wb') as file:
            pickle.dump(sbert_dist_val_embeddings, file)
        print("SentenceBERT embedding file saved.")
    with open(sbert_dist_val_embeddings_fname, 'rb') as file:
        sbert_dist_val_embeddings = pickle.load(file)
    sbert_avg_embedding = sbert_dist_val_embeddings.apply(lambda x: np.mean(x, axis = 0))

    matching_idx_dict = {}
    sample_list = sbert_utils.generate_pre_parallel(train, sbert_dist_val_embeddings, sbert_avg_embedding, ratio_list, lb_list, coverage_thres, sample_size, n_proc)
    aggre_dict = sbert_utils.get_matching_rows_parallel(train, sample_list, sbert_dist_val_embeddings, sbert_avg_embedding, n_proc)

    pre_list, keys_to_delete = [], []
    for k, v in aggre_dict.items():
        if len(v) / len(train) < coverage_thres:
            keys_to_delete.append(k)
        else:
            pre_list.append(k)

    for key in keys_to_delete:
        aggre_dict.pop(key)
        
    for pre in pre_list:
        matching_idx_dict[pre] = aggre_dict[pre]
        
    rule_list = sbert_utils.compute_cohenh_parallel(train, sbert_dist_val_embeddings, matching_idx_dict, pre_list, ub_list, n_proc)
    return rule_list