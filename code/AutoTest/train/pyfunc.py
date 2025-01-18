from util import pyfunc_utils

DFT_TYPE_LIST = ['url', 'ip', 'date', 'email']
DFT_RATIO_LIST = [0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99]

def rule_generate(train, params):   
    type_list = params['type_list'] if 'type_list' in params else DFT_TYPE_LIST
    ratio_list = params['ratio_list'] if 'ratio_list' in params else DFT_RATIO_LIST
    
    matching_idx_dict = {}
    pre_list = []
    
    for t in type_list:
        for ratio in ratio_list:
            pre = ('pyfunc', t, ratio)
            pre_list.append(pre)
            matching_idx_dict[tuple(pre)] = pyfunc_utils.get_matching_rows(train, pre).index.to_list()
            
    rule_list = pyfunc_utils.compute_cohen_h(train, matching_idx_dict, pre_list)
    return rule_list