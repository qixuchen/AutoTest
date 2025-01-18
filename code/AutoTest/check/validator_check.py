import pandas as pd
import numpy as np
import regex as re
import validators
import multiprocessing as mp
from util import utils, pyfunc_utils

def get_farthest_val_and_score(dist_val, t):
    assert t in ['email', 'ip', 'url']
    if t == 'email':
        for val in dist_val:
            if not validators.email(val):
                return val, 0
    elif t == 'ip':
        for val in dist_val:
            if not validators.ip_address.ipv4(val):
                return val, 0
    elif t == 'url':
        for val in dist_val:
            if not validators.url(val):
                return val, 0
    return None, 0


def validator_check(df, test_matching_dict, rule_list):
    results = []
    for i in range(len(rule_list)):
        if i % 5 == 0: print(f'{i}/{len(rule_list)}')
        pre, constraint, cohenh, conf, contingency = rule_list[i]
        assert constraint[0] == 'validator'
        t = constraint[1]
        
        if tuple(pre) not in test_matching_dict.keys(): continue
        matching_rows = utils.get_matching_rows_from_idx_dict(df, test_matching_dict, pre).copy()
        if len(matching_rows) == 0: continue
        matching_rows = matching_rows.assign(outlier = None, pre = None, outlier_score = -100, conf = -100, thres = -100, cohenh = -100, contingency = None)
        matching_rows[['outlier', 'outlier_score']] = matching_rows.apply(lambda row: get_farthest_val_and_score(row['dist_val'], t), axis = 1, result_type = 'expand')
        matching_rows = matching_rows[matching_rows['outlier'].notnull()].copy()
        if len(matching_rows) == 0: continue
        matching_rows['conf'] = conf
        matching_rows[['thres', 'cohenh']] = 0, cohenh
        matching_rows['pre'] = [pre for _ in range(len(matching_rows))]
        matching_rows['rule'] = [rule_list[i] for _ in range(len(matching_rows))]
        matching_rows['contingency'] = [contingency for _ in range(len(matching_rows))]
        results.append(matching_rows)
    return results

def validator_check_parallel_core(ns, start, end, queue):
        df = pd.DataFrame(ns.df[start : end], columns = ns.df_col, index = ns.df_idx[start : end])
        rule_list = ns.rule_list
        test_matching_dict = ns.test_matching_dict
        for i in range(len(rule_list)):
            if i % 5 == 0: print(f'{i}/{len(rule_list)}')
            pre, constraint, cohenh, conf, contingency = rule_list[i]
            assert constraint[0] == 'validator'
            t = constraint[1]
            
            if tuple(pre) not in test_matching_dict.keys(): continue
            matching_rows = utils.get_matching_rows_from_idx_dict(df, test_matching_dict, pre).copy()
            if len(matching_rows) == 0: continue
            matching_rows = matching_rows.assign(outlier = None, pre = None, outlier_score = -100, conf = -100, thres = -100, cohenh = -100, contingency = None)
            matching_rows[['outlier', 'outlier_score']] = matching_rows.apply(lambda row: get_farthest_val_and_score(row['dist_val'], t), axis = 1, result_type = 'expand')
            matching_rows = matching_rows[matching_rows['outlier'].notnull()].copy()
            if len(matching_rows) == 0: continue
            matching_rows['conf'] = conf
            matching_rows[['thres', 'cohenh']] = 0, cohenh
            matching_rows['pre'] = [pre for _ in range(len(matching_rows))]
            matching_rows['rule'] = [rule_list[i] for _ in range(len(matching_rows))]
            matching_rows['contingency'] = [contingency for _ in range(len(matching_rows))]
            queue.put(matching_rows)

def validator_check_parallel(df, test_matching_dict, rule_list, n_proc):
    with mp.Manager() as manager:
        ns = manager.Namespace()
        ns.df = manager.list(df.values.tolist())
        ns.df_idx = manager.list(df.index.tolist())
        ns.df_col = df.columns
        ns.rule_list = rule_list
        ns.test_matching_dict = test_matching_dict
        with mp.Pool() as pool:
            start_list = [len(df) * i // n_proc for i in range(n_proc)]
            end_list = [len(df) * (i + 1) // n_proc for i in range(n_proc)]
            queue_list = [mp.Manager().Queue() for _ in range(n_proc)]
            pool.starmap(validator_check_parallel_core, zip([ns] * n_proc, start_list, end_list, queue_list))
            results = []
            for q in queue_list:
                while not q.empty(): 
                    results.append(q.get())
            return results
        
        
def get_all_outliers(dist_val, t):
    assert t in ['email', 'ip', 'url']
    outliers = []
    if t == 'email':
        for val in dist_val:
            if not validators.email(val):
                outliers.append(val)
    elif t == 'ip':
        for val in dist_val:
            if not validators.ip_address.ipv4(val):
                outliers.append(val)
    elif t == 'url':
        for val in dist_val:
            if not validators.url(val):
                outliers.append(val)
    return None, 0


def validator_all_outliers(df, test_matching_dict, rule_list):
    results = []
    for i in range(len(rule_list)):
        if i % 5 == 0: print(f'{i}/{len(rule_list)}')
        pre, constraint, cohenh, conf, contingency = rule_list[i]
        assert constraint[0] == 'validator'
        t = constraint[1]
        
        if tuple(pre) not in test_matching_dict.keys(): continue
        matching_rows = utils.get_matching_rows_from_idx_dict(df, test_matching_dict, pre).copy()
        if len(matching_rows) == 0: continue
        matching_rows = matching_rows.assign(outlier = None, pre = None, outlier_score = -100, conf = -100, thres = -100, cohenh = -100, contingency = None)
        matching_rows['outlier'] = matching_rows.apply(lambda row: get_all_outliers(row['dist_val'], t), axis = 1)
        matching_rows = matching_rows[matching_rows['outlier'].apply(lambda x: len(x) > 0)].copy()
        if len(matching_rows) == 0: continue
        matching_rows['conf'] = conf
        matching_rows[['thres', 'cohenh']] = 0, cohenh
        matching_rows['pre'] = [pre for _ in range(len(matching_rows))]
        matching_rows['rule'] = [rule_list[i] for _ in range(len(matching_rows))]
        matching_rows['contingency'] = [contingency for _ in range(len(matching_rows))]
        results.append(matching_rows)
    return results