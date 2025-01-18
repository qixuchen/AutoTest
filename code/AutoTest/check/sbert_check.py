import pandas as pd
import numpy as np
import multiprocessing as mp
from util import utils, sbert_utils

def get_farthest_val_and_score(dist_val, dist_embeddings, ref_embed):
    distance_list = sbert_utils.dist_embeddings_to_ref(dist_embeddings, ref_embed)
    farthest_idx = np.argmax(distance_list)
    return dist_val[farthest_idx], distance_list[farthest_idx]

def sbert_check_parallel_core(ns, start, end, queue):
        df = pd.DataFrame(ns.df[start : end], columns = ns.df_col, index = ns.df_idx[start : end])
        sbert_dist_val_embeddings = pd.Series(ns.sbert_dist_val_embeddings[start : end], index = ns.sbert_dist_val_embeddings_idx[start : end])
        df = pd.concat([df, pd.DataFrame(sbert_dist_val_embeddings, columns=['sbert_dist_val_embeddings'])], axis=1)
        rule_list = ns.rule_list
        test_matching_dict = ns.test_matching_dict
        for i in range(len(rule_list)):
            if i % 100 == 0: print(f'{i}/{len(rule_list)}')
            pre, constraint, cohenh, conf, contingency = rule_list[i]
            assert constraint[0] == 'sbert'
            dist_thres = constraint[1]
            ref = pre[2]
            if tuple(pre) not in test_matching_dict.keys(): continue
            matching_rows = utils.get_matching_rows_from_idx_dict(df, test_matching_dict, pre)
            matching_rows = matching_rows[matching_rows['dist_val'].apply(lambda x: any(v == pre[2] for v in x))].copy()
            if len(matching_rows) == 0: continue
            matching_rows = matching_rows.assign(outlier = None, pre = None, outlier_score = -100, conf = -100, thres = -100, cohenh = -100, contingency = None)
            matching_rows
            ref_embed = sbert_utils.decide_embedding(ref)
            matching_rows[['outlier', 'outlier_score']] = matching_rows.apply(lambda row: get_farthest_val_and_score(row['dist_val'], row['sbert_dist_val_embeddings'], ref_embed), axis = 1, result_type = 'expand')
            matching_rows = matching_rows[matching_rows['outlier_score'] >= dist_thres].copy()
            if len(matching_rows) == 0: continue
            matching_rows['conf'] = conf
            matching_rows[['thres', 'cohenh']] = dist_thres, cohenh
            matching_rows['pre'] = [pre for _ in range(len(matching_rows))]
            matching_rows['rule'] = [rule_list[i] for _ in range(len(matching_rows))]
            matching_rows['contingency'] = [contingency for _ in range(len(matching_rows))]
            matching_rows.drop('sbert_dist_val_embeddings', axis=1, inplace=True)
            queue.put(matching_rows)

def sbert_check_parallel(df, test_matching_dict, rule_list, n_proc, sbert_dist_val_embeddings):
    with mp.Manager() as manager:
        ns = manager.Namespace()
        ns.df = manager.list(df.values.tolist())
        ns.df_idx = manager.list(df.index.tolist())
        ns.df_col = df.columns
        ns.rule_list = rule_list
        ns.test_matching_dict = test_matching_dict
        ns.sbert_dist_val_embeddings = manager.list(sbert_dist_val_embeddings.values.tolist())
        ns.sbert_dist_val_embeddings_idx = manager.list(sbert_dist_val_embeddings.index.tolist())
        with mp.Pool() as pool:
            start_list = [len(df) * i // n_proc for i in range(n_proc)]
            end_list = [len(df) * (i + 1) // n_proc for i in range(n_proc)]
            queue_list = [mp.Manager().Queue() for _ in range(n_proc)]
            pool.starmap(sbert_check_parallel_core, zip([ns] * n_proc, start_list, end_list, queue_list))
            results = []
            for q in queue_list:
                while not q.empty(): 
                    results.append(q.get())
            return results
        

def get_all_outliers(dist_val, dist_embeddings, ref_embed, dist_thres):
    distance_list = sbert_utils.dist_embeddings_to_ref(dist_embeddings, ref_embed)
    outlier_idx = np.where(np.array(distance_list) >= dist_thres)[0]
    if len(outlier_idx) == 0:
        return []
    else:
        return [dist_val[idx] for idx in outlier_idx]
    
def sbert_all_outliers_parallel_core(ns, start, end, queue):
        df = pd.DataFrame(ns.df[start : end], columns = ns.df_col, index = ns.df_idx[start : end])
        sbert_dist_val_embeddings = pd.Series(ns.sbert_dist_val_embeddings[start : end], index = ns.sbert_dist_val_embeddings_idx[start : end])
        df = pd.concat([df, pd.DataFrame(sbert_dist_val_embeddings, columns=['sbert_dist_val_embeddings'])], axis=1)
        rule_list = ns.rule_list
        test_matching_dict = ns.test_matching_dict
        for i in range(len(rule_list)):
            if i % 100 == 0: print(f'{i}/{len(rule_list)}')
            pre, constraint, cohenh, conf, contingency = rule_list[i]
            assert constraint[0] == 'sbert'
            dist_thres = constraint[1]
            ref = pre[2]
            if tuple(pre) not in test_matching_dict.keys(): continue
            matching_rows = utils.get_matching_rows_from_idx_dict(df, test_matching_dict, pre)
            matching_rows = matching_rows[matching_rows['dist_val'].apply(lambda x: any(v == pre[2] for v in x))].copy()
            if len(matching_rows) == 0: continue
            matching_rows = matching_rows.assign(outlier = None, pre = None, outlier_score = -100, conf = -100, thres = -100, cohenh = -100, contingency = None)
            matching_rows
            ref_embed = sbert_utils.decide_embedding(ref)
            matching_rows['outlier'] = matching_rows.apply(lambda row: get_all_outliers(row['dist_val'], row['sbert_dist_val_embeddings'], ref_embed, dist_thres), axis = 1)
            matching_rows = matching_rows[matching_rows['outlier'].apply(lambda x: len(x) > 0)].copy()
            if len(matching_rows) == 0: continue
            matching_rows['conf'] = conf
            matching_rows[['thres', 'cohenh']] = dist_thres, cohenh
            matching_rows['pre'] = [pre for _ in range(len(matching_rows))]
            matching_rows['rule'] = [rule_list[i] for _ in range(len(matching_rows))]
            matching_rows['contingency'] = [contingency for _ in range(len(matching_rows))]
            matching_rows.drop('sbert_dist_val_embeddings', axis=1, inplace=True)
            queue.put(matching_rows)

def sbert_all_outliers_parallel(df, test_matching_dict, rule_list, n_proc, sbert_dist_val_embeddings):
    with mp.Manager() as manager:
        ns = manager.Namespace()
        ns.df = manager.list(df.values.tolist())
        ns.df_idx = manager.list(df.index.tolist())
        ns.df_col = df.columns
        ns.rule_list = rule_list
        ns.test_matching_dict = test_matching_dict
        ns.sbert_dist_val_embeddings = manager.list(sbert_dist_val_embeddings.values.tolist())
        ns.sbert_dist_val_embeddings_idx = manager.list(sbert_dist_val_embeddings.index.tolist())
        with mp.Pool() as pool:
            start_list = [len(df) * i // n_proc for i in range(n_proc)]
            end_list = [len(df) * (i + 1) // n_proc for i in range(n_proc)]
            queue_list = [mp.Manager().Queue() for _ in range(n_proc)]
            pool.starmap(sbert_all_outliers_parallel_core, zip([ns] * n_proc, start_list, end_list, queue_list))
            results = []
            for q in queue_list:
                while not q.empty(): 
                    results.append(q.get())
            return results