import pandas as pd
import numpy as np
import regex as re
import multiprocessing as mp
from util import utils, pattern_utils

def get_farthest_val_and_score(dist_val, pattern):
    for val in dist_val:
        if re.match(pattern, val) == None:
            return val, 0
    return None, 0


def pattern_check(df, test_matching_dict, rule_list):
    results = []
    for i in range(len(rule_list)):
        if i % 5 == 0: print(f'{i}/{len(rule_list)}')
        pre, constraint, cohenh, conf, contingency = rule_list[i]
        assert constraint[0] == 'pattern'
        pattern = pre[3]
        
        if tuple(pre) not in test_matching_dict.keys(): continue
        matching_rows = utils.get_matching_rows_from_idx_dict(df, test_matching_dict, pre).copy()
        if len(matching_rows) == 0: continue
        matching_rows = matching_rows.assign(outlier = None, pre = None, outlier_score = -100, conf = -100, thres = -100, cohenh = -100, contingency = None)
        matching_rows[['outlier', 'outlier_score']] = matching_rows.apply(lambda row: get_farthest_val_and_score(row['dist_val'], pattern), axis = 1, result_type = 'expand')
        matching_rows = matching_rows[matching_rows['outlier'].notnull()].copy()
        if len(matching_rows) == 0: continue
        matching_rows['conf'] = conf
        matching_rows[['thres', 'cohenh']] = 0, cohenh
        matching_rows['pre'] = [pre for _ in range(len(matching_rows))]
        matching_rows['rule'] = [rule_list[i] for _ in range(len(matching_rows))]
        matching_rows['contingency'] = [contingency for _ in range(len(matching_rows))]
        results.append(matching_rows)
    return results


def get_all_outliers(dist_val, pattern):
    outliers = []
    for val in dist_val:
        if re.match(pattern, val) == None:
            outliers.append(val)
    return outliers


def pattern_all_outliers(df, test_matching_dict, rule_list):
    results = []
    for i in range(len(rule_list)):
        if i % 5 == 0: print(f'{i}/{len(rule_list)}')
        pre, constraint, cohenh, conf, contingency = rule_list[i]
        assert constraint[0] == 'pattern'
        pattern = pre[3]
        
        if tuple(pre) not in test_matching_dict.keys(): continue
        matching_rows = utils.get_matching_rows_from_idx_dict(df, test_matching_dict, pre).copy()
        if len(matching_rows) == 0: continue
        matching_rows = matching_rows.assign(outlier = None, pre = None, outlier_score = -100, conf = -100, thres = -100, cohenh = -100, contingency = None)
        matching_rows['outlier'] = matching_rows.apply(lambda row: get_all_outliers(row['dist_val'], pattern), axis = 1)
        matching_rows = matching_rows[matching_rows['outlier'].apply(lambda x: len(x) > 0)].copy()
        if len(matching_rows) == 0: continue
        matching_rows['conf'] = conf
        matching_rows[['thres', 'cohenh']] = 0, cohenh
        matching_rows['pre'] = [pre for _ in range(len(matching_rows))]
        matching_rows['rule'] = [rule_list[i] for _ in range(len(matching_rows))]
        matching_rows['contingency'] = [contingency for _ in range(len(matching_rows))]
        results.append(matching_rows)
    return results