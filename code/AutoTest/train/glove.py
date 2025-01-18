from util import embedding_utils

DFT_COV_THRES = 0.0003
DFT_SAMPLE_SIZE = 50
DFT_RATIO_LIST = [0.8]
DFT_LB_LIST = [3 + 0.5 * i for i in range(9)]
DFT_UB_LIST = [5 + 0.5 * i for i in range(15)]
DFT_NPROC = 48

def rule_generate(train, params):
    
    coverage_thres = params['coverage_thres'] if 'coverage_thres' in params else DFT_COV_THRES
    sample_size = params['sample_size'] if 'sample_size' in params else DFT_SAMPLE_SIZE
    ratio_list = params['ratio_list'] if 'ratio_list' in params else DFT_RATIO_LIST
    lb_list = params['lb_list'] if 'lb_list' in params else DFT_LB_LIST
    ub_list = params['ub_list'] if 'ub_list' in params else DFT_UB_LIST
    n_proc = params['n_proc'] if 'n_proc' in params else DFT_NPROC
    
    matching_idx_dict = {}
    sample_list = embedding_utils.generate_pre_parallel(train, ratio_list, lb_list, coverage_thres, sample_size, n_proc)
    aggre_dict = embedding_utils.get_matching_rows_parallel(train, sample_list, n_proc = n_proc)

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
        
    rule_list = embedding_utils.compute_cohenh_parallel(train, matching_idx_dict, pre_list, ub_list, n_proc)
    return rule_list