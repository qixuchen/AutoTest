import re
import pandas as pd
import numpy as np
from util import utils

def pattern_matching_ratio(dist_val, pattern):
    if len(dist_val) == 0: return 0
    return sum([bool(re.match(pattern, val)) for val in dist_val]) / len(dist_val)

def get_matching_rows(df, pre):
    assert pre[0] == 'pattern'
    if len(df) == 0:
        return df
    pattern, ratio = pre[3], pre[4]
    df = df[df['dist_val'].apply(lambda x: pattern_matching_ratio(x, pattern) >= ratio)]
    return df

def get_matching_rows_pre_list(df, pre_list):
    matching_idx_dict = {}
    if len(df) == 0: return matching_idx_dict
    for i in range(len(pre_list)):
        pre = pre_list[i]
        assert pre[0] == 'pattern'
        if i % 20 == 0: print(f'{i}/{len(pre_list)}')
        pattern, ratio = pre[3], pre[4]
        matching_rows = df[df['dist_val'].apply(lambda x: pattern_matching_ratio(x, pattern) >= ratio)]
        matching_idx_dict[tuple(pre)] = matching_rows.index.to_list()
    return matching_idx_dict

def compute_cohen_h(train: pd.DataFrame, matching_idx_dict: dict, pre_list: list):
    results = []
    for i in range(len(pre_list)):
        pre = pre_list[i]
        assert pre[0] == 'pattern'
        precond = utils.get_matching_rows_from_idx_dict(train, matching_idx_dict, pre)
        outdom = train.loc[~train.index.isin(precond.index)]
        pattern = pre[3]

        indom_trigger = precond['dist_val'].apply(lambda x: any(re.match(pattern, val) == None for val in x)).sum()
        indom_not_trigger = len(precond) - indom_trigger
        outdom_trigger = outdom['dist_val'].apply(lambda x: any(re.match(pattern, val) == None for val in x)).sum()
        outdom_not_trigger = len(outdom) - outdom_trigger
        
        ch = utils.cohen_h(indom_trigger, indom_not_trigger, outdom_trigger, outdom_not_trigger)
        conf = utils.estimate_confidence(indom_not_trigger, indom_trigger)
        results.append([tuple(pre), ('pattern', 0), ch, conf, [indom_trigger, indom_not_trigger, outdom_trigger, outdom_not_trigger]])
    return results