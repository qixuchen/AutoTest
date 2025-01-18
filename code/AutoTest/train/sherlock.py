import os
import pandas as pd
from util import sherlock_utils


DFT_COV_THRES = 0.0003
DFT_RATIO_LIST = [0.8, 0.9, 0.95]
DFT_SCORE_BARS = [0.1 + 0.05 * i for i in range(15)]
DFT_THRES_LIST = [0, 0.002, 0.005, 0.007, 0.01, 0.02, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3]
DFT_INTM_RES_DIR = os.path.join(os.getcwd(), 'sherlock_intermediate_results')
DFT_MIN_LABEL_SCORE_FNAME = 'sherlock_min_label_scores.pickle'
DFT_NPROC = 32

def rule_generate(train, params):
    coverage_thres = params['coverage_thres'] if 'coverage_thres' in params else DFT_COV_THRES
    ratio_list = params['ratio_list'] if 'ratio_list' in params else DFT_RATIO_LIST
    score_bars = params['score_bars'] if 'score_bars' in params else DFT_SCORE_BARS
    thres_list = params['thres_list'] if 'thres_list' in params else DFT_THRES_LIST
    intermediate_result_dir = params['intermediate_result_dir'] if 'intermediate_result_dir' in params else DFT_INTM_RES_DIR
    min_label_score_fname = params['min_label_score_fname'] if 'min_label_score_fname' in params else DFT_MIN_LABEL_SCORE_FNAME
    n_proc = params['n_proc'] if 'n_proc' in params else DFT_NPROC
    
    if not os.path.exists(os.path.join(intermediate_result_dir, min_label_score_fname)):
        print("Sherlock preprocessing result not found, computing ...")
        print(f'Results will be saved to {os.path.join(intermediate_result_dir, min_label_score_fname)}.')
        if not os.path.exists(intermediate_result_dir):
            os.makedirs(intermediate_result_dir)
        sherlock_utils.min_score_in_each_label_parallel(train, intermediate_result_dir, min_label_score_fname, n_proc)
    min_scores = pd.read_pickle(os.path.join(intermediate_result_dir, min_label_score_fname))
    
    matching_idx_dict = {}
    pre_list = []
    filter_dict = sherlock_utils.build_filter_dict_parallel(train, n_proc)

    for label in sherlock_utils.class_list:
        for ratio in ratio_list:
            for score in score_bars:
                pre = ['cta', label, ratio, score]
                pre_list.append(pre)
    aggre_dict = sherlock_utils.get_matching_rows_parallel(train, pre_list, filter_dict, n_proc)
    
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
        
    rule_list = sherlock_utils.compute_cohenh_parallel(train, matching_idx_dict, pre_list, thres_list, min_scores, n_proc)
    return rule_list