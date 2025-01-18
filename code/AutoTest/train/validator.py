from util import validator_utils

DFT_TYPE_LIST = ['ip', 'email', 'url']
DFT_RATIO_LIST = [0.8, 0.9]
DFT_NPROC = 32

def rule_generate(train, params):
    type_list = params['type_list'] if 'type_list' in params else DFT_TYPE_LIST
    ratio_list = params['ratio_list'] if 'ratio_list' in params else DFT_RATIO_LIST
    n_proc = params['n_proc'] if 'n_proc' in params else DFT_NPROC
    
    matching_idx_dict = {}
    pre_list = []

    for t in type_list:
        for ratio in ratio_list:
            pre = ('validator', t, ratio)
            pre_list.append(pre)
    aggre_dict = validator_utils.get_matching_rows_parallel(train, pre_list, n_proc)
    
    for pre in pre_list:
        matching_idx_dict[pre] = aggre_dict[pre]
        
    rule_list = validator_utils.compute_cohen_h(train, matching_idx_dict, pre_list)
    return rule_list