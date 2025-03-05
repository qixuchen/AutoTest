from config import config
import os, sys
sys.path.append(os.path.join(config.dir.project_base_dir, 'doduo-project'))
import argparse
import torch
import pandas as pd
import numpy as np
import multiprocessing as mp
from itertools import chain
from math import ceil
from doduo import Doduo
from util import utils
from func import load_corpus

torch.set_num_threads(4)

class_list = [
    "address", "affiliate", "affiliation", "age", "album", "area", "artist",
    "birthDate", "birthPlace", "brand", "capacity", "category", "city",
    "class", "classification", "club", "code", "collection", "command",
    "company", "component", "continent", "country", "county", "creator",
    "credit", "currency", "day", "depth", "description", "director",
    "duration", "education", "elevation", "family", "fileSize", "format",
    "gender", "genre", "grades", "isbn", "industry", "jockey", "language",
    "location", "manufacturer", "name", "nationality", "notes", "operator",
    "order", "organisation", "origin", "owner", "person", "plays", "position",
    "product", "publisher", "range", "rank", "ranking", "region", "religion",
    "requirement", "result", "sales", "service", "sex", "species", "state",
    "status", "symbol", "team", "teamName", "type", "weight", "year"
]

PRECOND_CUTOFF = 3
global_doduo_model = None

def load_doduo():
    # Load Doduo model
    args = argparse.Namespace
    args.model = "viznet" # or args.model = "viznet"
    return Doduo(args, basedir=os.path.join(config.dir.project_base_dir, 'doduo-project'))

def validate_doduo():
    global global_doduo_model
    if global_doduo_model is None:
        global_doduo_model = load_doduo()

def predict(dist_val: list, model = None):
    global global_doduo_model
    if model is None:
        validate_doduo()
        model = global_doduo_model
    df = pd.DataFrame(dist_val)
    annot_df = model.annotate_columns(df)
    y_pred = annot_df.predvec.detach().numpy()[0]
    label_idx = annot_df.coltypeidx.item()
    label_score = annot_df.labelscore.item()
    predicted_label = annot_df.coltypes[0]
    return y_pred, label_idx, label_score, predicted_label

def predict_multi_row(df: pd.DataFrame, col_name: str = 'dist_val', model = None):
    results = df.apply(lambda x: predict(x[col_name], model = model), axis = 1, result_type='expand')
    y_pred = results[0].to_list()
    label_idx = results[1].to_numpy()
    label_score = results[2].to_numpy()
    predicted_label = results[3].to_list()
    return np.array(y_pred), label_idx, label_score, predicted_label

def get_matching_rows(df, pre, dist_val_scores):
    assert pre[0] == 'doduo'
    label, ratio, score_bar = pre[1:4]
    df = pd.concat([df, pd.DataFrame(dist_val_scores, columns=['dist_val_scores'])], axis=1).copy()
    df = df[df['dist_val'].apply(lambda x: len([v for v in x if not utils.contains_non_english_chars(v)]) >= 0.8 * len(x))]
    if len(df) == 0: return df
    label_idx = class_list.index(label)
    matching_rows = df
    sorted_row_scores = matching_rows['dist_val_scores'].apply(lambda x: sorted([score[label_idx] for score in x[0]], reverse = True))
    score_at_ratio = sorted_row_scores.apply(lambda x: x[ceil(len(x) * ratio) - 1])
    matching_rows = matching_rows[score_at_ratio >= score_bar]
    return matching_rows

def get_matching_rows_parallel_core(ns, start, end, queue):
    df = pd.DataFrame(ns.df[start : end], columns = ns.df_col, index = ns.df_idx[start : end])
    dist_val_scores = pd.Series(ns.dist_val_scores[start : end], index = ns.dist_val_scores_idx[start : end])
    df = pd.concat([df, pd.DataFrame(dist_val_scores, columns=['dist_val_scores'])], axis=1)
    df = df[df['dist_val'].apply(lambda x: len([v for v in x if not utils.contains_non_english_chars(v)]) >= 0.8 * len(x))]
    if len(df) == 0: return
    pre_list = ns.pre_list
    idx_dict = {}
    for label in class_list:
        if label == 'name': continue 
        label_idx = class_list.index(label)
        matching_rows = df
        sorted_row_scores = matching_rows['dist_val_scores'].apply(lambda x: sorted([score[label_idx] for score in x[0]], reverse = True))
        pre_sub_list = [pre for pre in pre_list if pre[1] == label]
        for pre in pre_sub_list:
            ratio, score_bar = pre[2], pre[3]
            score_at_ratio = sorted_row_scores.apply(lambda x: x[ceil(len(x) * ratio) - 1])
            matching_rows = matching_rows[score_at_ratio >= score_bar]
            idx_dict[tuple(pre)] = matching_rows.index.to_list()
    queue.put(idx_dict)

def get_matching_rows_parallel(df, pre_list, dist_val_scores, n_proc):
    for pre in pre_list:
        assert pre[0] == 'doduo'
    with mp.Manager() as manager:
        ns = manager.Namespace()
        ns.df = manager.list(df.values.tolist())
        ns.df_idx = manager.list(df.index.tolist())
        ns.df_col = df.columns
        ns.pre_list = pre_list
        ns.dist_val_scores = manager.list(dist_val_scores.values.tolist())
        ns.dist_val_scores_idx = manager.list(dist_val_scores.index.tolist())
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

def dist_val_scores_parallel_core(ns, start, end, dir, result_fname, order):
    df = pd.DataFrame(ns.df[start : end], columns = ns.df_col, index = ns.df_idx[start : end])
    for i in range(ceil(len(df) / 1000)):
        print(f"{result_fname}_proc_{order}_seg_{i}")
        fname = os.path.join(dir, f"{result_fname}_proc_{order}_seg_{i}.pkl")
        seg = df.iloc[i * 1000 : (i+1) * 1000]
        min_scores = seg['dist_val'].apply(lambda x: predict_multi_row(pd.DataFrame({'dist_val': [[val] for val in x]})))
        min_scores.to_pickle(fname)

def dist_val_scores_parallel(df, dir, result_fname, n_proc):
    with mp.Manager() as manager:
        ns = manager.Namespace()
        ns.df = manager.list(df.values.tolist())
        ns.df_idx = manager.list(df.index.tolist())
        ns.df_col = df.columns
        with mp.Pool() as pool:
            start_list = [len(df) * i // n_proc for i in range(n_proc)]
            end_list = [len(df) * (i + 1) // n_proc for i in range(n_proc)]
            pool.starmap(dist_val_scores_parallel_core, zip([ns] * n_proc, start_list, end_list, [dir] * n_proc, [result_fname] * n_proc, range(n_proc)))
            
    dist_val_scores = pd.DataFrame()
    for order in range(n_proc):
        for i in range(ceil(len(df) / (n_proc * 1000))):
            seg_fname = os.path.join(dir, f"{result_fname}_proc_{order}_seg_{i}.pkl")
            dist_val_scores = pd.concat([dist_val_scores, pd.read_pickle(seg_fname)])
    dist_val_scores[0].to_pickle(os.path.join(dir, result_fname))     

def compute_cohenh_parallel_core(ns, start, end, queue):
    df = pd.DataFrame(ns.df[start : end], columns = ns.df_col, index = ns.df_idx[start : end])
    dist_val_scores = pd.Series(ns.dist_val_scores[start : end], index = ns.dist_val_scores_idx[start : end])
    pre_list = ns.pre_list
    thres_list = ns.thres_list
    matching_idx_dict = ns.matching_idx_dict
    for i in range(len(pre_list)):
        if i % 10 == 0: print(f"Progress: {i} / {len(pre_list)}")
        pre = pre_list[i]
        assert pre[0] == 'doduo'
        precond = utils.get_matching_rows_from_idx_dict(df, matching_idx_dict, pre)
        label = pre[1]
        label_idx = class_list.index(label)
        for score_thres in thres_list:
            indom_scores = dist_val_scores.loc[dist_val_scores.index.isin(precond.index)]
            indom_label_score = indom_scores.apply(lambda x: min([score[label_idx] for score in x[0]]))
            outdom_scores = dist_val_scores.loc[~dist_val_scores.index.isin(precond.index)]
            outdom_label_score = outdom_scores.apply(lambda x: min([score[label_idx] for score in x[0]]))

            indom_trigger = (indom_label_score <= score_thres).sum()
            indom_not_trigger = len(indom_label_score) - indom_trigger
            outdom_trigger = (outdom_label_score <= score_thres).sum()
            outdom_not_trigger = len(outdom_label_score) - outdom_trigger
            queue.put([tuple(pre) + ('doduo', score_thres), [indom_trigger, indom_not_trigger, outdom_trigger, outdom_not_trigger]])
    
def compute_cohenh_parallel(df, matching_idx_dict, pre_list, thres_list, dist_val_scores, n_proc):
    with mp.Manager() as manager:
        ns = manager.Namespace()
        ns.df = manager.list(df.values.tolist())
        ns.df_idx = manager.list(df.index.tolist())
        ns.df_col = df.columns
        ns.dist_val_scores = manager.list(dist_val_scores.values.tolist())
        ns.dist_val_scores_idx = manager.list(dist_val_scores.index.tolist())
        ns.pre_list = manager.list(pre_list)
        ns.thres_list = manager.list(thres_list)
        ns.matching_idx_dict = manager.dict(matching_idx_dict)
        with mp.Pool() as pool:
            start_list = [len(df) * i // n_proc for i in range(n_proc)]
            end_list = [len(df) * (i + 1) // n_proc for i in range(n_proc)]
            queue_list = [mp.Manager().Queue() for _ in range(n_proc)]
            pool.starmap(compute_cohenh_parallel_core, zip([ns] * n_proc, start_list, end_list, queue_list))

            results_dict = {}
            for q in queue_list:
                while not q.empty(): 
                    item = q.get()
                    if item[0] not in results_dict:
                        results_dict[item[0]] = item[1]
                    else:
                        for i in range(len(item[1])):
                            results_dict[item[0]][i] += item[1][i]
            results = []
            for k, v in results_dict.items():
                ch = utils.cohen_h(*v)
                conf = utils.estimate_confidence(v[1], v[0])
                results.append([k[:-2], k[-2:], ch, conf, v])
        return results
    
# def build_score_dict(precond, label, model = None):
#     dist_val_list = list(set(chain(*precond['dist_val'].tolist())))
#     df = pd.DataFrame({'dist_val': [[val] for val in dist_val_list]})
#     y_pred, _, _, _ = predict_multi_row(df, model = model)
#     label_idx = class_list.index(label)
#     label_score = y_pred[:, label_idx]
#     score_dict = {}
#     for i in range(len(dist_val_list)):
#         score_dict[dist_val_list[i]] = label_score[i]
#     return score_dict
    
# def build_filter_dict(df, model = None):
#     filter_dict = {}
#     df = df[df['dist_val'].apply(lambda x: len([v for v in x if not utils.contains_non_english_chars(v)]) >= 0.8 * len(x))]
#     if len(df) == 0: return filter_dict
#     _, _, label_score, predicted_label = predict_multi_row(df, model = model)
#     all_labels = list(set(predicted_label))
#     df_idx = df.index.to_list()
#     for label in all_labels:
#         if label == 'name': continue #seems doduo is overly confident on name
#         matching_idx = [df_idx[i] for i in range(len(predicted_label)) if predicted_label[i] == label and label_score[i] > PRECOND_CUTOFF]
#         filter_dict[label] = matching_idx
#     return filter_dict

# def build_filter_dict_parallel_core(ns, start, end, queue):
#     filter_dict = {}
#     df = pd.DataFrame(ns.df[start : end], columns = ns.df_col, index = ns.df_idx[start : end])
#     df = df[df['dist_val'].apply(lambda x: len([v for v in x if not utils.contains_non_english_chars(v)]) >= 0.8 * len(x))]
#     if len(df) == 0: 
#         queue.put(filter_dict)
#         return
#     _, _, label_score, predicted_label = predict_multi_row(df, model = None)
#     all_labels = list(set(predicted_label))
#     df_idx = df.index.to_list()
#     for label in all_labels:
#         if label == 'name': continue #seems doduo is overly confident on name
#         matching_idx = [df_idx[i] for i in range(len(predicted_label)) if predicted_label[i] == label and label_score[i] > PRECOND_CUTOFF]
#         filter_dict[label] = matching_idx
#     queue.put(filter_dict)

# def build_filter_dict_parallel(df, n_proc):
#     with mp.Manager() as manager:
#         ns = manager.Namespace()
#         ns.df = manager.list(df.values.tolist())
#         ns.df_idx = manager.list(df.index.tolist())
#         ns.df_col = df.columns
#         with mp.Pool() as pool:
#             start_list = [len(df) * i // n_proc for i in range(n_proc)]
#             end_list = [len(df) * (i + 1) // n_proc for i in range(n_proc)]
#             queue_list = [mp.Manager().Queue() for _ in range(n_proc)]
#             pool.starmap(build_filter_dict_parallel_core, zip([ns] * n_proc, start_list, end_list, queue_list))
#             results = []
#             for q in queue_list:
#                 while not q.empty(): 
#                     results.append(q.get())
#             filter_dict = {}
#             for r in results:
#                 for k, v in r.items():
#                     if k not in filter_dict.keys():
#                         filter_dict[k] = v
#                     else:
#                         filter_dict[k] = filter_dict[k] + v
#             return filter_dict

# def min_score_in_each_label(dist_val, model = None):
#     min_scores = np.full(len(class_list), 100, dtype = np.float32)
#     y_pred, _, _, _ = predict_multi_row(pd.DataFrame({'dist_val': [[val] for val in dist_val]}), model = model)
#     for label_idx in range(len(class_list)):
#         min_scores[label_idx] = np.min(y_pred[:, label_idx])
#     return min_scores

# def min_score_in_each_label_parallel_core(ns, start, end, order):
#     df = pd.DataFrame(ns.df[start : end], columns = ns.df_col, index = ns.df_idx[start : end])
#     for i in range(ceil(len(df) / 1000)):
#         print(f"{load_corpus.CORPUS_NAME}_min_scores_doduo_proc_{order}_seg_{i}")
#         fname = f"./output/doduo/{load_corpus.CORPUS_NAME}_min_scores_doduo_proc_{order}_seg_{i}.pickle"
#         seg = df.iloc[i * 1000 : (i+1) * 1000]
#         min_scores = seg['dist_val'].apply(lambda x: min_score_in_each_label(x))
#         min_scores.to_pickle(fname)

# def min_score_in_each_label_parallel(df, n_proc):
#     with mp.Manager() as manager:
#         ns = manager.Namespace()
#         ns.df = manager.list(df.values.tolist())
#         ns.df_idx = manager.list(df.index.tolist())
#         ns.df_col = df.columns
#         with mp.Pool() as pool:
#             start_list = [len(df) * i // n_proc for i in range(n_proc)]
#             end_list = [len(df) * (i + 1) // n_proc for i in range(n_proc)]
#             pool.starmap(min_score_in_each_label_parallel_core, zip([ns] * n_proc, start_list, end_list, range(n_proc)))