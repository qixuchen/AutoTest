import pandas as pd
import numpy as np
import multiprocessing as mp
from util import utils, doduo_utils

def get_farthest_val_and_score(dist_val, dist_val_scores, label):
    label_idx = doduo_utils.class_list.index(label)
    label_score = dist_val_scores[0][:, label_idx]
    farthest_idx = np.argmin(label_score)
    return dist_val[farthest_idx], label_score[farthest_idx]

def doduo_check_parallel_core(ns, start, end, queue):
        df = pd.DataFrame(ns.df[start : end], columns = ns.df_col, index = ns.df_idx[start : end])
        doduo_dist_val_scores = pd.Series(ns.doduo_dist_val_scores[start : end], index = ns.doduo_dist_val_scores_idx[start : end])
        df = pd.concat([df, pd.DataFrame(doduo_dist_val_scores, columns=['doduo_dist_val_scores'])], axis=1)
        rule_list = ns.rule_list
        test_matching_dict = ns.test_matching_dict
        for i in range(len(rule_list)):
            if i % 10 == 0: print(f'{i}/{len(rule_list)}')
            pre, constraint, cohenh, conf, contingency = rule_list[i]
            assert constraint[0] == 'doduo'
            score_thres = constraint[1]
            label = pre[1]
            
            if tuple(pre) not in test_matching_dict.keys(): continue
            matching_rows = utils.get_matching_rows_from_idx_dict(df, test_matching_dict, pre).copy()
            if len(matching_rows) == 0: continue
            matching_rows = matching_rows.assign(outlier = None, pre = None, outlier_score = -100, conf = -100, thres = -100, cohenh = -100, contingency = None)
            matching_rows[['outlier', 'outlier_score']] = matching_rows.apply(lambda row: get_farthest_val_and_score(row['dist_val'], row['doduo_dist_val_scores'], label), axis = 1, result_type = 'expand')
            matching_rows = matching_rows[matching_rows['outlier_score'] <= score_thres].copy()
            if len(matching_rows) == 0: continue
            matching_rows['conf'] = conf
            matching_rows[['thres', 'cohenh']] = score_thres, cohenh
            matching_rows['pre'] = [pre for _ in range(len(matching_rows))]
            matching_rows['rule'] = [rule_list[i] for _ in range(len(matching_rows))]
            matching_rows['contingency'] = [contingency for _ in range(len(matching_rows))]
            matching_rows.drop('doduo_dist_val_scores', axis=1, inplace=True)
            queue.put(matching_rows)


def doduo_check_parallel(df, test_matching_dict, rule_list, n_proc, doduo_dist_val_scores):
    with mp.Manager() as manager:
        ns = manager.Namespace()
        ns.df = manager.list(df.values.tolist())
        ns.df_idx = manager.list(df.index.tolist())
        ns.df_col = df.columns
        ns.rule_list = rule_list
        ns.test_matching_dict = test_matching_dict
        ns.doduo_dist_val_scores = manager.list(doduo_dist_val_scores.values.tolist())
        ns.doduo_dist_val_scores_idx = manager.list(doduo_dist_val_scores.index.tolist())
        with mp.Pool() as pool:
            start_list = [len(df) * i // n_proc for i in range(n_proc)]
            end_list = [len(df) * (i + 1) // n_proc for i in range(n_proc)]
            queue_list = [mp.Manager().Queue() for _ in range(n_proc)]
            pool.starmap(doduo_check_parallel_core, zip([ns] * n_proc, start_list, end_list, queue_list))
            results = []
            for q in queue_list:
                while not q.empty(): 
                    results.append(q.get())
            return results
        
        
def get_all_outliers(dist_val, dist_val_scores, label, score_thres):
    label_idx = doduo_utils.class_list.index(label)
    label_score = dist_val_scores[0][:, label_idx]
    outlier_idx = np.where(label_score <= score_thres)[0]
    if len(outlier_idx) == 0:
        return []
    else:
        return [dist_val[idx] for idx in outlier_idx]
    

def doduo_all_outliers_parallel_core(ns, start, end, queue):
    df = pd.DataFrame(ns.df[start : end], columns = ns.df_col, index = ns.df_idx[start : end])
    doduo_dist_val_scores = pd.Series(ns.doduo_dist_val_scores[start : end], index = ns.doduo_dist_val_scores_idx[start : end])
    df = pd.concat([df, pd.DataFrame(doduo_dist_val_scores, columns=['doduo_dist_val_scores'])], axis=1)
    rule_list = ns.rule_list
    test_matching_dict = ns.test_matching_dict
    for i in range(len(rule_list)):
        if i % 10 == 0: print(f'{i}/{len(rule_list)}')
        pre, constraint, cohenh, conf, contingency = rule_list[i]
        assert constraint[0] == 'doduo'
        score_thres = constraint[1]
        label = pre[1]
        
        if tuple(pre) not in test_matching_dict.keys(): continue
        matching_rows = utils.get_matching_rows_from_idx_dict(df, test_matching_dict, pre).copy()
        if len(matching_rows) == 0: continue
        matching_rows = matching_rows.assign(outlier = None, pre = None, outlier_score = -100, conf = -100, thres = -100, cohenh = -100, contingency = None)
        matching_rows['outlier'] = matching_rows.apply(lambda row: get_all_outliers(row['dist_val'], row['doduo_dist_val_scores'], label, score_thres), axis = 1)
        matching_rows = matching_rows[matching_rows['outlier'].apply(lambda x: len(x) > 0)].copy()
        if len(matching_rows) == 0: continue
        matching_rows['conf'] = conf
        matching_rows[['thres', 'cohenh']] = score_thres, cohenh
        matching_rows['pre'] = [pre for _ in range(len(matching_rows))]
        matching_rows['rule'] = [rule_list[i] for _ in range(len(matching_rows))]
        matching_rows['contingency'] = [contingency for _ in range(len(matching_rows))]
        matching_rows.drop('doduo_dist_val_scores', axis=1, inplace=True)
        queue.put(matching_rows)


def doduo_all_outliers_parallel(df, test_matching_dict, rule_list, n_proc, doduo_dist_val_scores):
    with mp.Manager() as manager:
        ns = manager.Namespace()
        ns.df = manager.list(df.values.tolist())
        ns.df_idx = manager.list(df.index.tolist())
        ns.df_col = df.columns
        ns.rule_list = rule_list
        ns.test_matching_dict = test_matching_dict
        ns.doduo_dist_val_scores = manager.list(doduo_dist_val_scores.values.tolist())
        ns.doduo_dist_val_scores_idx = manager.list(doduo_dist_val_scores.index.tolist())
        with mp.Pool() as pool:
            start_list = [len(df) * i // n_proc for i in range(n_proc)]
            end_list = [len(df) * (i + 1) // n_proc for i in range(n_proc)]
            queue_list = [mp.Manager().Queue() for _ in range(n_proc)]
            pool.starmap(doduo_all_outliers_parallel_core, zip([ns] * n_proc, start_list, end_list, queue_list))
            results = []
            for q in queue_list:
                while not q.empty(): 
                    results.append(q.get())
            return results