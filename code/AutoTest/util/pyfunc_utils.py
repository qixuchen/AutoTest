import re
import pandas as pd
import numpy as np
import multiprocessing as mp
from util import utils
from dataprep import clean

def is_email_col(dist_val, ratio):
    if any([utils.contains_non_alphabet(val) for val in dist_val]): return False
    return len([val for val in dist_val if clean.validate_email(val)]) >= ratio * len(dist_val)

def is_ip_col(dist_val, ratio):
    if any([utils.contains_non_alphabet(val) for val in dist_val]): return False
    return len([val for val in dist_val if clean.validate_ip(val)]) >= ratio * len(dist_val)

def is_url_col(dist_val, ratio):
    numeric_pattern = r'^[\+\-\d\.,]+$'
    if any([utils.contains_non_alphabet(val) for val in dist_val]): return False
    if sum([bool(re.match(numeric_pattern, val)) for val in dist_val]) > 0.5 * len(dist_val): return False
    return len([val for val in dist_val if clean.validate_url(val)]) >= ratio * len(dist_val)

def is_date_col(dist_val, ratio):
    numeric_pattern = r'^[\+\-\d\.,]+$'
    if any([utils.contains_non_alphabet(val) for val in dist_val]): return False
    if sum([bool(re.match(numeric_pattern, val)) for val in dist_val]) > 0.5 * len(dist_val): return False
    return len([val for val in dist_val if clean.validate_date(val)]) >= ratio * len(dist_val)

def validate(val, t):
    assert t in ['email', 'ip', 'url', 'date']
    if t == 'email':
        return clean.validate_email(val) 
    elif t == 'ip':
        return clean.validate_ip(val) 
    elif t == 'url':
        return clean.validate_url(val) 
    elif t == 'date':
        return clean.validate_date(val) 
    
def get_matching_rows(df, pre):
    assert pre[0] == 'pyfunc' and pre[1] in ['email', 'ip', 'url', 'date']
    ratio = pre[2]
    if len(df) == 0:
        return df
    if pre[1] == 'email':
        return df[df['dist_val'].apply(lambda x: is_email_col(x, ratio))]
    elif pre[1] == 'ip':
        return df[df['dist_val'].apply(lambda x: is_ip_col(x, ratio))]
    elif pre[1] == 'url':
        return df[df['dist_val'].apply(lambda x: is_url_col(x, ratio))]
    elif pre[1] == 'date':
        return df[df['dist_val'].apply(lambda x: is_date_col(x, ratio))]
    
def get_matching_rows_pre_list(df, pre_list):
    matching_idx_dict = {}
    if len(df) == 0: return matching_idx_dict
    for pre in pre_list:
        assert pre[0] == 'pyfunc' and pre[1] in ['email', 'ip', 'url', 'date']
        matching_rows = get_matching_rows(df, pre)
        matching_idx_dict[tuple(pre)] = matching_rows.index.to_list()
    return matching_idx_dict

def get_matching_rows_parallel_core(ns, start, end, queue):
        df = pd.DataFrame(ns.df[start : end], columns = ns.df_col, index = ns.df_idx[start : end])
        pre_list = ns.pre_list
        idx_dict = {}
        for i in range(len(pre_list)):
            pre = pre_list[i]
            print(pre)
            matching_rows = get_matching_rows(df, pre)
            idx_dict[tuple(pre)] = matching_rows.index.to_list()
        queue.put(idx_dict)

def get_matching_rows_parallel(df, pre_list, n_proc):
    with mp.Manager() as manager:
        ns = manager.Namespace()
        ns.df = manager.list(df.values.tolist())
        ns.df_idx = manager.list(df.index.tolist())
        ns.df_col = df.columns
        ns.pre_list = pre_list
        with mp.Pool() as pool:
            start_list = [len(df) * i // n_proc for i in range(n_proc)]
            end_list = [len(df) * (i + 1) // n_proc for i in range(n_proc)]
            queue_list = [mp.Manager().Queue() for _ in range(n_proc)]
            pool.starmap(get_matching_rows_parallel_core, zip([ns] * n_proc, start_list, end_list, queue_list))
            results = []
            for q in queue_list:
                while not q.empty(): 
                    results.append(q.get())
            aggre_dict = {}
            for d in results:
                for k, v in d.items():
                    if k not in aggre_dict.keys():
                        aggre_dict[k] = v[:]
                    else:
                        aggre_dict[k] = aggre_dict[k] + v
            return aggre_dict

def compute_cohenh_email(df: pd.DataFrame, matching_idx_dict: dict, pre: list):
    assert pre[0] == 'pyfunc' and pre[1] == 'email'

    def not_all_email(dist_val):
        if any([utils.contains_non_alphabet(val) for val in dist_val]): return True
        for val in dist_val:
            if not clean.validate_email(val):
                return True
        return False
    
    precond = utils.get_matching_rows_from_idx_dict(df, matching_idx_dict, pre)
    outdom = df.loc[~df.index.isin(precond.index)]

    indom_trigger = precond['dist_val'].apply(not_all_email).sum()
    indom_not_trigger = len(precond) - indom_trigger
    outdom_trigger = outdom['dist_val'].apply(not_all_email).sum()
    outdom_not_trigger = len(outdom) - outdom_trigger

    ch = utils.cohen_h(indom_trigger, indom_not_trigger, outdom_trigger, outdom_not_trigger)
    return ch, [indom_trigger, indom_not_trigger, outdom_trigger, outdom_not_trigger]

def compute_cohenh_ip(df: pd.DataFrame, matching_idx_dict: dict, pre: list):
    assert pre[0] == 'pyfunc' and pre[1] == 'ip'

    def not_all_ip(dist_val):
        if any([utils.contains_non_alphabet(val) for val in dist_val]): return True
        for val in dist_val:
            if not clean.validate_ip(val):
                return True
        return False
    
    precond = utils.get_matching_rows_from_idx_dict(df, matching_idx_dict, pre)
    outdom = df.loc[~df.index.isin(precond.index)]

    indom_trigger = precond['dist_val'].apply(not_all_ip).sum()
    indom_not_trigger = len(precond) - indom_trigger
    outdom_trigger = outdom['dist_val'].apply(not_all_ip).sum()
    outdom_not_trigger = len(outdom) - outdom_trigger

    ch = utils.cohen_h(indom_trigger, indom_not_trigger, outdom_trigger, outdom_not_trigger)
    return ch, [indom_trigger, indom_not_trigger, outdom_trigger, outdom_not_trigger]

def compute_cohenh_url(df: pd.DataFrame, matching_idx_dict: dict, pre: list):
    assert pre[0] == 'pyfunc' and pre[1] == 'url'

    def not_all_url(dist_val):
        if any([utils.contains_non_alphabet(val) for val in dist_val]): return True
        for val in dist_val:
            if not clean.validate_url(val):
                return True
        return False
    
    precond = utils.get_matching_rows_from_idx_dict(df, matching_idx_dict, pre)
    outdom = df.loc[~df.index.isin(precond.index)]

    indom_trigger = precond['dist_val'].apply(not_all_url).sum()
    indom_not_trigger = len(precond) - indom_trigger
    outdom_trigger = outdom['dist_val'].apply(not_all_url).sum()
    outdom_not_trigger = len(outdom) - outdom_trigger

    ch = utils.cohen_h(indom_trigger, indom_not_trigger, outdom_trigger, outdom_not_trigger)
    return ch, [indom_trigger, indom_not_trigger, outdom_trigger, outdom_not_trigger]

def compute_cohenh_date(df: pd.DataFrame, matching_idx_dict: dict, pre: list):
    assert pre[0] == 'pyfunc' and pre[1] == 'date'

    def not_all_date(dist_val):
        if any([utils.contains_non_alphabet(val) for val in dist_val]): return True
        for val in dist_val:
            if not clean.validate_date(val):
                return True
        return False
    
    precond = utils.get_matching_rows_from_idx_dict(df, matching_idx_dict, pre)
    outdom = df.loc[~df.index.isin(precond.index)]

    indom_trigger = precond['dist_val'].apply(not_all_date).sum()
    indom_not_trigger = len(precond) - indom_trigger
    outdom_trigger = outdom['dist_val'].apply(not_all_date).sum()
    outdom_not_trigger = len(outdom) - outdom_trigger

    ch = utils.cohen_h(indom_trigger, indom_not_trigger, outdom_trigger, outdom_not_trigger)
    return ch, [indom_trigger, indom_not_trigger, outdom_trigger, outdom_not_trigger]


def compute_cohen_h(df: pd.DataFrame, matching_idx_dict: dict, pre_list: list):
    results = []
    for i in range(len(pre_list)):
        pre = pre_list[i]
        assert pre[0] == 'pyfunc' and pre[1] in ['email', 'ip', 'url', 'date']
        if pre[1] == 'email':
            ch, contingency = compute_cohenh_email(df, matching_idx_dict, pre)
        elif pre[1] == 'ip':
            ch, contingency = compute_cohenh_ip(df, matching_idx_dict, pre)
        elif pre[1] == 'url':
            ch, contingency = compute_cohenh_url(df, matching_idx_dict, pre)
        elif pre[1] == 'date':
            ch, contingency = compute_cohenh_date(df, matching_idx_dict, pre)
        
        conf = utils.estimate_confidence(contingency[1], contingency[0])
        results.append([tuple(pre), tuple(pre[:2]), ch, conf, contingency])
    return results