from config import config
import os, sys
sys.path.append(os.path.join(config.dir.project_base_dir, 'sherlock-project'))
import pandas as pd
import numpy as np
import pyarrow as pa
import multiprocessing as mp
from io import StringIO
from sherlock import helpers
from sherlock.deploy.model import SherlockModel
from sherlock.functional import extract_features_to_csv
from sherlock.features.paragraph_vectors import initialise_pretrained_model, initialise_nltk
from sherlock.features.preprocessing import (
    extract_features,
    extract_features_non_write_to_file,
    convert_string_lists_to_lists,
    prepare_feature_extraction,
    load_parquet_values,
)
from sherlock.features.word_embeddings import initialise_word_embeddings
from itertools import chain
from math import ceil, floor
from func import load_corpus
import util.utils as utils

class_list = ['address', 'affiliate', 'affiliation', 'age', 'album', 'area', 'artist',
 'birth Date', 'birth Place', 'brand', 'capacity', 'category', 'city', 'class',
 'classification', 'club', 'code', 'collection', 'command', 'company',
 'component', 'continent', 'country', 'county', 'creator', 'credit', 'currency',
 'day', 'depth', 'description', 'director', 'duration', 'education', 'elevation',
 'family', 'file Size', 'format', 'gender', 'genre', 'grades', 'industry', 'isbn',
 'jockey', 'language', 'location', 'manufacturer', 'name', 'nationality',
 'notes', 'operator', 'order', 'organisation', 'origin', 'owner', 'person',
 'plays', 'position', 'product', 'publisher', 'range', 'rank', 'ranking',
 'region', 'religion', 'requirement', 'result', 'sales', 'service', 'sex',
 'species', 'state', 'status', 'symbol', 'team', 'team Name', 'type', 'weight',
 'year']

PRECOND_CUTOFF = 0.1
global_sherlock_model = None


def initialize_sherlock():
    prepare_feature_extraction()
    initialise_word_embeddings()
    initialise_pretrained_model(400)
    initialise_nltk()
    sherlock_model = SherlockModel()
    sherlock_model.initialize_model_from_json(with_weights=True, model_id="sherlock") ###
    return sherlock_model

def validate_sherlock():
    global global_sherlock_model
    if global_sherlock_model is None:
        global_sherlock_model = initialize_sherlock()

def predict(data: pd.Series, model = None, verbose = False):
    global global_sherlock_model
    if model is None:
        validate_sherlock()
        model = global_sherlock_model
    feature_vectors = pd.read_csv(StringIO(extract_features_non_write_to_file(data)), dtype=np.float32)
    y_pred, score, predicted_labels = model.predict(feature_vectors, "sherlock")
    y_pred = np.nan_to_num(y_pred, nan = 0) # replace all nan with 0
    score = np.nan_to_num(score, nan = 0)
    return y_pred, score, predicted_labels 

def predict_label_score(data: pd.Series, label: str, model = None):
    y_pred, _, _ = predict([[d] for d in data], model = model)
    label_idx = class_list.index(label)
    label_score = y_pred[:, label_idx]
    return label_score

def build_score_dict(precond, label, model = None):
    dist_val_list = list(set(chain(*precond['dist_val'].tolist())))
    val_ll = [[v] for v in dist_val_list]
    y_pred, _, _ = predict(val_ll, model = model)
    label_idx = class_list.index(label)
    label_score = y_pred[:, label_idx]
    score_dict = {}
    for i in range(len(val_ll)):
        score_dict[val_ll[i][0]] = label_score[i]
    return score_dict

def rows_with_ratio_val_gt_score_bar(df, label, ratio, score_bar, model = None):
    if len(df) == 0: return df
    score_dict = build_score_dict(df, label, model = model)
    row_scores = df['dist_val'].apply(lambda x: sorted([score_dict[v] for v in x], reverse = True))
    score_at_ratio = row_scores.apply(lambda x: x[ceil(len(x) * ratio) - 1])
    df = df[score_at_ratio >= score_bar]
    return df

def build_filter_dict(df, model = None):
    filter_dict = {}
    # df = df[df['dist_val'].apply(lambda x: any(utils.contains_non_alphabet(v) or utils.contain_digit(v) for v in x) == False)]
    df = df[df['dist_val'].apply(lambda x: len([v for v in x if not utils.contains_non_english_chars(v)]) >= 0.8 * len(x))]
    if len(df) == 0: return filter_dict
    y_pred, score, predicted_labels = predict(df['dist_val'], model = model)
    all_labels = list(set(predicted_labels))
    df_idx = df.index.to_list()
    for label in all_labels:
        if label == 'address': continue# address is the default class if all score are nan
        matching_idx = [df_idx[i] for i in range(len(predicted_labels)) if predicted_labels[i] == label and score[i] > PRECOND_CUTOFF]
        filter_dict[label] = matching_idx
    return filter_dict

def build_filter_dict_parallel_core(ns, start, end, queue):
    filter_dict = {}
    df = pd.DataFrame(ns.df[start : end], columns = ns.df_col, index = ns.df_idx[start : end])
    df = df[df['dist_val'].apply(lambda x: len([v for v in x if not utils.contains_non_english_chars(v)]) >= 0.8 * len(x))]
    if len(df) == 0: 
        queue.put(filter_dict)
        return
    _, score, predicted_labels = predict(df['dist_val'], model = None)
    all_labels = list(set(predicted_labels))
    df_idx = df.index.to_list()
    for label in all_labels:
        if label == 'address': continue # address is the default class if all score are nan
        matching_idx = [df_idx[i] for i in range(len(predicted_labels)) if predicted_labels[i] == label and score[i] > PRECOND_CUTOFF]
        filter_dict[label] = matching_idx
    queue.put(filter_dict)

def build_filter_dict_parallel(df, n_proc):
    with mp.Manager() as manager:
        ns = manager.Namespace()
        ns.df = manager.list(df.values.tolist())
        ns.df_idx = manager.list(df.index.tolist())
        ns.df_col = df.columns
        with mp.Pool() as pool:
            start_list = [len(df) * i // n_proc for i in range(n_proc)]
            end_list = [len(df) * (i + 1) // n_proc for i in range(n_proc)]
            queue_list = [mp.Manager().Queue() for _ in range(n_proc)]
            pool.starmap(build_filter_dict_parallel_core, zip([ns] * n_proc, start_list, end_list, queue_list))
            results = []
            for q in queue_list:
                while not q.empty(): 
                    results.append(q.get())
            filter_dict = {}
            for r in results:
                for k, v in r.items():
                    if k not in filter_dict.keys():
                        filter_dict[k] = v
                    else:
                        filter_dict[k] = filter_dict[k] + v
            return filter_dict

def min_score_in_each_label(dist_val: pd.Series, model = None):
    min_scores = np.full(len(class_list), 100, dtype = np.float32)
    y_pred, _, _ = predict([[v] for v in dist_val], model = model)
    for label_idx in range(len(class_list)):
        min_scores[label_idx] = np.min(y_pred[:, label_idx])
    return min_scores

def min_score_in_each_label_parallel_core(ns, start, end, intermediate_result_dir, order):
    df = pd.DataFrame(ns.df[start : end], columns = ns.df_col, index = ns.df_idx[start : end])
    for i in range(ceil(len(df) / 1000)):
        print(f"{load_corpus.CORPUS_NAME}_min_scores_sherlock_proc_{order}_seg_{i}")
        fname = os.path.join(intermediate_result_dir, f'{load_corpus.CORPUS_NAME}_min_scores_sherlock_proc_{order}_seg_{i}.pickle')
        seg = df.iloc[i * 1000 : (i+1) * 1000]
        min_scores = seg['dist_val'].apply(lambda x: min_score_in_each_label(x))
        min_scores.to_pickle(fname)

def min_score_in_each_label_parallel(df, intermediate_result_dir, result_fname, n_proc):
    with mp.Manager() as manager:
        ns = manager.Namespace()
        ns.df = manager.list(df.values.tolist())
        ns.df_idx = manager.list(df.index.tolist())
        ns.df_col = df.columns
        with mp.Pool() as pool:
            start_list = [len(df) * i // n_proc for i in range(n_proc)]
            end_list = [len(df) * (i + 1) // n_proc for i in range(n_proc)]
            pool.starmap(min_score_in_each_label_parallel_core, zip([ns] * n_proc, start_list, end_list, [intermediate_result_dir] * n_proc, range(n_proc)))
            
    min_scores = pd.DataFrame()
    for order in range(n_proc):
        for i in range(ceil(len(df) / (n_proc * 1000))):
            seg_fname = os.path.join(intermediate_result_dir, f'{load_corpus.CORPUS_NAME}_min_scores_sherlock_proc_{order}_seg_{i}.pickle')
            min_scores = pd.concat([min_scores, pd.read_pickle(seg_fname)])
    min_scores = min_scores[0]
    min_scores.to_pickle(os.path.join(intermediate_result_dir, result_fname))
        
def get_matching_rows(df, pre, filter_dict):
    assert pre[0] == 'cta'
    label, ratio, score_bar = pre[1], pre[2], pre[3]
    matching_idx = filter_dict[label]
    matching_rows = df.loc[df.index.isin(matching_idx)]
    if len(matching_rows) == 0: return matching_rows
    matching_rows = matching_rows[matching_rows['dist_val'].apply(lambda x: len([v for v in x if not utils.contains_non_english_chars(v)]) >= 0.8 * len(x))]
    if len(matching_rows) == 0: return matching_rows
    matching_rows = rows_with_ratio_val_gt_score_bar(matching_rows, label, ratio, score_bar)
    return matching_rows
        
def get_matching_rows_parallel_core(ns, start, end, queue):
    df = pd.DataFrame(ns.df[start : end], columns = ns.df_col, index = ns.df_idx[start : end])
    pre_list = ns.pre_list
    filter_dict = ns.filter_dict
    idx_dict = {}
    for label, matching_idx in filter_dict.items():
        matching_rows = df.loc[df.index.isin(matching_idx)]
        if len(matching_rows) == 0: continue
        matching_rows = matching_rows[matching_rows['dist_val'].apply(lambda x: len([v for v in x if not utils.contains_non_english_chars(v)]) >= 0.8 * len(x))]
        if len(matching_rows) == 0: continue
        score_dict = build_score_dict(matching_rows, label, model = None)
        sorted_row_scores = matching_rows['dist_val'].apply(lambda x: sorted([score_dict[v] for v in x], reverse = True))
        pre_sub_list = [pre for pre in pre_list if pre[1] == label]
        for pre in pre_sub_list:
            ratio, score_bar = pre[2], pre[3]
            score_at_ratio = sorted_row_scores.apply(lambda x: x[ceil(len(x) * ratio) - 1])
            matching_rows = matching_rows[score_at_ratio >= score_bar]
            idx_dict[tuple(pre)] = matching_rows.index.to_list()
    queue.put(idx_dict)

def get_matching_rows_parallel(df, pre_list, filter_dict, n_proc):
    with mp.Manager() as manager:
        ns = manager.Namespace()
        ns.df = manager.list(df.values.tolist())
        ns.df_idx = manager.list(df.index.tolist())
        ns.df_col = df.columns
        ns.pre_list = pre_list
        ns.filter_dict = filter_dict
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
        
def compute_cohenh(train: pd.DataFrame, matching_idx_dict: dict, pre: list, min_scores: pd.Series, score_thres: float):
    assert pre[0] == 'cta' 
    label = pre[1]
    label_idx = class_list.index(label)
    precond = utils.get_matching_rows_from_idx_dict(train, matching_idx_dict, pre)

    indom_scores = min_scores.loc[min_scores.index.isin(precond.index)]
    indom_label_score = indom_scores.apply(lambda x: x[label_idx])
    outdom_scores = min_scores.loc[~min_scores.index.isin(precond.index)]
    outdom_label_score = outdom_scores.apply(lambda x: x[label_idx])

    indom_trigger = (indom_label_score <= score_thres).sum()
    indom_not_trigger = len(indom_label_score) - indom_trigger
    outdom_trigger = (outdom_label_score <= score_thres).sum()
    outdom_not_trigger = len(outdom_label_score) - outdom_trigger

    ch = utils.cohen_h(indom_trigger, indom_not_trigger, outdom_trigger, outdom_not_trigger)
    return ch, [indom_trigger, indom_not_trigger, outdom_trigger, outdom_not_trigger]

def compute_cohenh_parallel_core(ns, start, end, queue):
    df = pd.DataFrame(ns.df[start : end], columns = ns.df_col, index = ns.df_idx[start : end])
    min_scores = pd.Series(ns.min_scores[start : end], index = ns.min_scores_idx[start : end])
    pre_list = ns.pre_list
    thres_list = ns.thres_list
    matching_idx_dict = ns.matching_idx_dict
    for i in range(len(pre_list)):
        if i % 10 == 0: print(f"Progress: {i} / {len(pre_list)}")
        pre = pre_list[i]
        assert pre[0] == 'cta'
        precond = utils.get_matching_rows_from_idx_dict(df, matching_idx_dict, pre)
        label = pre[1]
        label_idx = class_list.index(label)
        for score_thres in thres_list:
            indom_scores = min_scores.loc[min_scores.index.isin(precond.index)]
            indom_label_score = indom_scores.apply(lambda x: x[label_idx])
            outdom_scores = min_scores.loc[~min_scores.index.isin(precond.index)]
            outdom_label_score = outdom_scores.apply(lambda x: x[label_idx])

            indom_trigger = (indom_label_score <= score_thres).sum()
            indom_not_trigger = len(indom_label_score) - indom_trigger
            outdom_trigger = (outdom_label_score <= score_thres).sum()
            outdom_not_trigger = len(outdom_label_score) - outdom_trigger
            queue.put([tuple(pre) + ('cta', score_thres), [indom_trigger, indom_not_trigger, outdom_trigger, outdom_not_trigger]])
    
def compute_cohenh_parallel(df, matching_idx_dict, pre_list, thres_list, min_scores, n_proc):
    with mp.Manager() as manager:
        ns = manager.Namespace()
        ns.df = manager.list(df.values.tolist())
        ns.df_idx = manager.list(df.index.tolist())
        ns.df_col = df.columns
        ns.min_scores = manager.list(min_scores.values.tolist())
        ns.min_scores_idx = manager.list(min_scores.index.tolist())
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

def baseline_rule_gen_core(ns, start, end, queue):
    pre_list = ns.pre_list[start : end]
    unique_val_list = ns.unique_val_list[start : end]
    for i in range(len(pre_list)):
        pre = pre_list[i]
        unique_vals = unique_val_list[i]
        _, class_name, _, _ = pre_list[0]
        scores = predict_label_score(unique_vals, class_name)
        outer_score = sorted(scores)[floor(0.025 * len(scores))]
        
        rule = []
        rule.append(tuple(pre))
        rule.append(('cta', outer_score))
        rule.append(0)
        rule.append(0)
        rule.append((0, 0, 0, 0))
        queue.put(rule)

def baseline_rule_gen(unique_val_list, pre_list, n_proc):
    with mp.Manager() as manager:
        ns = manager.Namespace()
        ns.pre_list = pre_list
        ns.unique_val_list = unique_val_list
        with mp.Pool() as pool:
            start_list = [len(pre_list) * i // n_proc for i in range(n_proc)]
            end_list = [len(pre_list) * (i + 1) // n_proc for i in range(n_proc)]
            queue_list = [mp.Manager().Queue() for _ in range(n_proc)]
            pool.starmap(baseline_rule_gen_core, zip([ns] * n_proc, start_list, end_list, queue_list))
            results = []
            for q in queue_list:
                while not q.empty(): 
                    results.append(q.get())
            return results