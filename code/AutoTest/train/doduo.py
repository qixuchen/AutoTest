import os
import pandas as pd
from util import doduo_utils

DFT_COV_THRES = 0.0003
DFT_RATIO_LIST = [0.8]
DFT_SCORE_BARS = [4, 4.5]
DFT_THRES_LIST = [-2.5, -2, -1.5, -1]
DFT_INTM_RES_DIR = os.path.join(os.getcwd(), 'doduo_intermediate_results')
DFT_VAL_SCORE_FNAME = 'doduo_dist_val_scores.pickle'
DFT_NPROC = 15


def rule_generate(train, params):
    coverage_thres = params['coverage_thres'] if 'coverage_thres' in params else DFT_COV_THRES
    ratio_list = params['ratio_list'] if 'ratio_list' in params else DFT_RATIO_LIST
    score_bars = params['score_bars'] if 'score_bars' in params else DFT_SCORE_BARS
    thres_list = params['thres_list'] if 'thres_list' in params else DFT_THRES_LIST
    intermediate_result_dir = params['intermediate_result_dir'] if 'intermediate_result_dir' in params else DFT_INTM_RES_DIR
    dist_val_scores_fname = params['dist_val_scores_fname'] if 'dist_val_scores_fname' in params else DFT_VAL_SCORE_FNAME
    n_proc = params['n_proc'] if 'n_proc' in params else DFT_NPROC
    
    if not os.path.exists(os.path.join(intermediate_result_dir, dist_val_scores_fname)):
        print("Doduo preprocessing result not found, computing ...")
        print(f'Results will be saved to {os.path.join(intermediate_result_dir, dist_val_scores_fname)}.')
        if not os.path.exists(intermediate_result_dir):
            os.makedirs(intermediate_result_dir)
        doduo_utils.dist_val_scores_parallel(train, intermediate_result_dir, dist_val_scores_fname, n_proc)
        print("Doduo preprocessing file saved.")
    dist_val_scores = pd.read_pickle(os.path.join(intermediate_result_dir, dist_val_scores_fname)) 
    
    matching_idx_dict = {}
    pre_list = []
    for label in doduo_utils.class_list:
        for ratio in ratio_list:
            for score in score_bars:
                pre = ['doduo', label, ratio, score]
                pre_list.append(pre)
    aggre_dict = doduo_utils.get_matching_rows_parallel(train, pre_list, dist_val_scores, n_proc)
    
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
        
    rule_list = doduo_utils.compute_cohenh_parallel(train, matching_idx_dict, pre_list, thres_list, dist_val_scores, n_proc)
    return rule_list