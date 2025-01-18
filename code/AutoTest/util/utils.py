import pandas as pd
import regex as re
import numpy as np
import scipy.stats as stats
from scipy import spatial
from collections import defaultdict
from itertools import chain
from statistics import mean
from util import embedding_utils, sbert_utils, sherlock_utils, doduo_utils, pattern_utils, pyfunc_utils, validator_utils
from math import ceil, sqrt, asin, log

epsilon = 1e-8

CONS_TYPE = 0
SIGN = 1
BOUND = 2
COV = 3
FPR = 4
REC = 5
PRE_TYPE = 6

def FPR(contingency_table):
    return contingency_table[0] / (contingency_table[0] + contingency_table[1])

def cohen_d(contingency_table):
    #               in_domain  out_domain
    # trigger           a           c
    # non_trig          b           d
    a, b, c, d = contingency_table
    m1 = a / (a + b)
    m2 = c / (c + d)
    var1 = a * (1 - m1) * (1 - m1)  + b * m1 * m1
    var2 = c * (1 - m2) * (1 - m2)  + d * m2 * m2
    s = sqrt( (var1 + var2) / (a + b + c + d - 2) )
    return (m2 - m1) / s

def binomial_conf_ub(contingency_table):
    a, b, _, _ = contingency_table
    z = 1.96
    n = a + b
    p = a / n
    return p + z * sqrt(a * b) / (n * sqrt(n))

def wilson_score_interval_FPR_upper_bound(n, fpr):
    if n == 0: return 1
    # if fpr == 0 : return 3 / n # rule of three
    z = 1.96
    t1 = fpr + z * z / (2 * n)
    t2 = 1 + z * z / n
    t3 = sqrt(fpr * (1 - fpr) / n + z * z / (4 * n * n))
    return t1 / t2 + z * t3 / t2

def estimate_confidence(num_in: int, num_out: int):
    if (num_in + num_out) == 0: return 1
    FPR = num_out / (num_in + num_out)
    return wilson_score_interval_FPR_upper_bound((num_in + num_out), FPR)

def chi2(in_trg, in_n_trg, out_trg, out_n_trg):
    f_obs = np.array([[in_trg, in_n_trg], [out_trg, out_n_trg]])
    return stats.chi2_contingency(f_obs)

def cohen_h(in_trg, in_n_trg, out_trg, out_n_trg):
    if (in_trg + in_n_trg) == 0 or (out_trg + out_n_trg) == 0: return 0
    p1 = in_trg / (in_trg + in_n_trg)
    p2 = out_trg / (out_trg + out_n_trg)
    return 2 * abs(asin(sqrt(p1)) - asin(sqrt(p2)))

def token_size(val: str):
    return len(re.findall(r'[A-Za-z]+', val))

def to_string(rule):
    return "   ".join([str(c) for c in rule])

def is_non_empty(val):
    return val not in [None, '', 'n/a', '-', '—', '–']

def remove_prefix(s: str):
    if s.startswith('__header__'):
        return s[len('__header__'):]
    else: 
        return s

def contain_digit(val):
    for char in val:
        if char.isdigit():
            return True
    return False

def contains_non_alphabet(val):
    for char in val:
        if not char.isascii():
            return True
    return False

def contains_non_english_chars(val):
    # pattern = r'[^a-zA-Z\s,\.\(\)]'
    pattern = r'[^a-zA-Z\s]'
    return bool(re.search(pattern, val))

def get_pre(rule):
    if rule[PRE_TYPE] in ['values', 'embed', 'pattern', 'sbert']:
        return rule[PRE_TYPE:]
    raise ValueError("Invalid pre-condition type.")

def get_matching_rows(df, pre, filter_dict_sherlock = None, filter_dict_doduo = None):
    assert pre[0] in ['cta', 'embed', 'doduo', 'sbert', 'pattern', 'pyfunc', 'validator']
    if len(df) == 0:
        return df
    if pre[0] == 'values':
        ref_length = pre[1]
        pre_values = pre[2:2 + ref_length]
        for w in pre_values:
            if len(df) == 0:
                break
            if w.startswith('__header__'):
                df = df[df['col_header'] == w[len('__header__'):]]
            else:
                df = df[df['dist_val'].apply(lambda x: any(val == w for val in x))]
        return df
    elif pre[0] == 'pattern':
        return pattern_utils.get_matching_rows(df, pre)
    elif pre[0] == 'embed':
        return embedding_utils.get_matching_rows(df, pre)
    elif pre[0] == 'sbert':
        return sbert_utils.get_matching_rows(df, pre)
    elif pre[0] == 'cta':
        assert filter_dict_sherlock is not None
        return sherlock_utils.get_matching_rows(df, pre, filter_dict_sherlock)
    elif pre[0] == 'doduo':
        assert filter_dict_doduo is not None
        return sherlock_utils.get_matching_rows(df, pre, filter_dict_doduo)
    elif pre[0] == 'pyfunc':
        return pyfunc_utils.get_matching_rows(df, pre)
    elif pre[0] == 'validator':
        return validator_utils.get_matching_rows(df, pre)

def get_matching_rows_from_ref_list(df, ref_list):
    for ref in ref_list:
        if len(df) == 0:
                break
        df = df[df['dist_val'].apply(lambda x: any(val == ref for val in x))]
    return df

def get_matching_rows_from_idx_dict(df, matching_idx_dict, pre):  
    assert tuple(pre) in matching_idx_dict.keys()
    return df.loc[df.index.isin(matching_idx_dict[tuple(pre)])]

def build_matching_idx_dict_from_pre_list(df, pre_list):
    matching_idx_dict = {}
    filter_dict_sherlock , filter_dict_doduo = None, None
    if any(pre[0] == 'cta' for pre in pre_list):
        filter_dict_sherlock = sherlock_utils.build_filter_dict(df)
    if any(pre[0] == 'doduo' for pre in pre_list):
        filter_dict_doduo = doduo_utils.build_filter_dict(df)
    for i in range(len(pre_list)):
        if i % 5 == 0: 
            print(i / len(pre_list))
        pre = pre_list[i]
        matching_rows = get_matching_rows(df, pre, filter_dict_sherlock, filter_dict_doduo)
        matching_idx_dict[tuple(pre)] = matching_rows.index.to_list()
    return matching_idx_dict

def update_matching_idx_dict_from_pre_list(df, matching_idx_dict, pre_list):
    filter_dict_sherlock , filter_dict_doduo = None, None
    if any(pre[0] == 'cta' for pre in pre_list):
        filter_dict_sherlock = sherlock_utils.build_filter_dict(df)
    if any(pre[0] == 'doduo' for pre in pre_list):
        filter_dict_doduo = doduo_utils.build_filter_dict(df)
    for i in range(len(pre_list)):
        if i % 5 == 0: 
            print(i / len(pre_list))
        pre = pre_list[i]
        matching_rows = get_matching_rows(df, pre, filter_dict_sherlock, filter_dict_doduo)
        matching_idx_dict[tuple(pre)] = matching_rows.index.to_list()
    return matching_idx_dict

def load_pre_list(precond_files: list):
    pre_list = []
    for f in precond_files:
        with open(f, 'r') as file:
            for line in file:
                pre = line.rstrip('\n').split('\t')
                if pre[0] == 'values':
                    try:
                        pre_length = int(pre[1]) + 2
                        if len(pre) == pre_length:
                            pre[1] = int(pre[1])
                    except ValueError:
                        raise ValueError('Invalid rule formulation.')
                elif pre[0] == 'cta' or pre[0] == 'doduo':
                    try:
                        pre[2] = float(pre[2])
                        pre[3] = float(pre[3])
                    except ValueError:
                        raise ValueError('Invalid rule formulation.')
                elif pre[0] == 'pattern':
                    try:
                        pre[1] = int(pre[1])
                        pre[4] = float(pre[4])
                    except ValueError:
                        raise ValueError('Invalid rule formulation.')
                elif pre[0] == 'embed' or pre[0] == 'sbert':
                    try:
                        pre[1] = int(pre[1])
                        pre[3] = float(pre[3])
                        pre[4] = float(pre[4])
                    except ValueError:
                        raise ValueError('Invalid rule formulation.')
                elif pre[0] == 'pyfunc' or pre[0] == 'validator':
                    try:
                        pre[2] = float(pre[2])
                    except ValueError:
                        raise ValueError('Invalid rule formulation.')
                else:
                    raise ValueError('Rule type cannot be recognized.')
                pre_list.append(pre)
    return pre_list

def get_covered_idx(corpus, matching_idx_dict, pre_list):
    slices = []
    for pre in pre_list:
        slices.append(get_matching_rows_from_idx_dict(corpus, matching_idx_dict, pre).index.tolist())
    covered_idx = set([element for sublist in slices for element in sublist])
    return covered_idx

def compute_coverage(corpus, matching_idx_dict, pre_list):
    covered_idx = get_covered_idx(corpus, matching_idx_dict, pre_list)
    covered = corpus.loc[corpus.index.isin(covered_idx)]
    return len(covered) / len(corpus)

def compute_uncovered_cols(corpus, matching_idx_dict, pre_list):
    covered_idx = get_covered_idx(corpus, matching_idx_dict, pre_list)
    uncovered_cols = corpus.loc[~corpus.index.isin(covered_idx)]
    return uncovered_cols

def build_matching_idx_dict_from_pre_list_parallel(df, pre_list, n_proc, sbert_dist_val_embeddings = None, doduo_dist_val_scores = None):
    matching_idx_dict = {}
    return update_matching_idx_dict_from_pre_list_parallel(df, matching_idx_dict, pre_list, n_proc, sbert_dist_val_embeddings, doduo_dist_val_scores)

def update_matching_idx_dict_from_pre_list_parallel(df, matching_idx_dict, pre_list, n_proc, sbert_dist_val_embeddings = None, doduo_dist_val_scores = None):
    assert all([pre[0] in ['cta', 'embed', 'doduo', 'sbert', 'pattern', 'pyfunc', 'validator'] for pre in pre_list]) 
    if any([pre[0] == 'cta' for pre in pre_list]):
        filter_dict = sherlock_utils.build_filter_dict_parallel(df, n_proc)
        print('filter dict built')
        sub_pre_list = [pre for pre in pre_list if pre[0] == 'cta']
        matching_idx_dict.update(sherlock_utils.get_matching_rows_parallel(df, sub_pre_list, filter_dict, n_proc))

    if any([pre[0] == 'doduo' for pre in pre_list]):
        n_proc_doduo = min(15, n_proc)
        sub_pre_list = [pre for pre in pre_list if pre[0] == 'doduo']
        matching_idx_dict.update(doduo_utils.get_matching_rows_parallel(df, sub_pre_list, doduo_dist_val_scores, n_proc_doduo))

    if any([pre[0] == 'embed' for pre in pre_list]):
        sub_pre_list = [pre for pre in pre_list if pre[0] == 'embed']
        matching_idx_dict.update(embedding_utils.get_matching_rows_parallel(df, sub_pre_list, n_proc))

    if any([pre[0] == 'sbert' for pre in pre_list]):
        n_proc_sbert = min(8, n_proc)
        if sbert_dist_val_embeddings is None:
            sbert_dist_val_embeddings = sbert_utils.dist_val_embeddings_parallel(df, n_proc_sbert)
        sbert_avg_embedding = sbert_dist_val_embeddings.apply(lambda x: np.mean(x, axis = 0))
        print('sbert embedding computed')
        sub_pre_list = [pre for pre in pre_list if pre[0] == 'sbert']
        matching_idx_dict.update(sbert_utils.get_matching_rows_parallel(df, sub_pre_list, sbert_dist_val_embeddings, sbert_avg_embedding, n_proc_sbert))

    if any([pre[0] == 'pattern' for pre in pre_list]):
        sub_pre_list = [pre for pre in pre_list if pre[0] == 'pattern']
        matching_idx_dict.update(pattern_utils.get_matching_rows_pre_list(df, sub_pre_list))

    if any([pre[0] == 'pyfunc' for pre in pre_list]):
        sub_pre_list = [pre for pre in pre_list if pre[0] == 'pyfunc']
        matching_idx_dict.update(pyfunc_utils.get_matching_rows_parallel(df, sub_pre_list, n_proc))

    if any([pre[0] == 'validator' for pre in pre_list]):
        sub_pre_list = [pre for pre in pre_list if pre[0] == 'validator']
        matching_idx_dict.update(validator_utils.get_matching_rows_parallel(df, sub_pre_list, n_proc))

    return matching_idx_dict