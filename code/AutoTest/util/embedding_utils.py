import os
import pandas as pd
import numpy as np
import regex as re
import multiprocessing as mp
import random
from config import config
from math import ceil
from util import utils
from statistics import mean
from scipy import spatial

dim = 50
glove_file = os.path.join(config.dir.storage_root_dir, config.dir.storage_root.glove, f'glove.6B.{dim}d.txt')
embeddings_dict = {}
oov_vector = np.full((dim,), -100)

def validate_embedding_dict():
    if len(embeddings_dict) == 0:
        load_glove_dict()

def load_glove_dict():
    global embeddings_dict
    with open(glove_file, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector

def decide_embedding(val):
    validate_embedding_dict()
    if val in embeddings_dict:
        return embeddings_dict[val]
    else:
        return oov_vector

def is_oov(val):
    validate_embedding_dict()
    return val not in embeddings_dict

def contain_oov(val):
    validate_embedding_dict()
    tokens = re.findall(r'[A-Za-z]+', val)
    if len(tokens) == 0:
        return True
    return any(np.array_equal(decide_embedding(val), oov_vector) for val in tokens)

def all_zero(val):
    return all(x == 0 for x in decide_embedding(val))

def avg_embedding(dist_val):
    filtered_embeddings = [decide_embedding(val) for val in dist_val if not contain_oov(val)]
    if len(filtered_embeddings) == 0:
        return oov_vector
    return np.mean(filtered_embeddings, axis = 0)

def pairwise_distance(val: str, ref: str) -> float:
    if not is_oov(val):
        return spatial.distance.euclidean(decide_embedding(val), decide_embedding(ref))
    tokens = re.findall(r'[A-Za-z]+', val)
    if len(tokens) == 0 or any(is_oov(t) for t in tokens): 
        return spatial.distance.euclidean(oov_vector, decide_embedding(ref)) 
    dist_list = [spatial.distance.euclidean(decide_embedding(t), decide_embedding(ref)) for t in tokens]
    return np.mean(dist_list, axis = 0)

def dist_to_ref(dist_val: list, ref: str) -> list:
    validate_embedding_dict()
    if len(dist_val) == 0:
        raise ValueError("No value exists")
    return [pairwise_distance(val, ref) for val in dist_val]


def max_dist_gt_thres(dist_val: list, ref: str, dist_thres) -> list:
    validate_embedding_dict()
    if len(dist_val) == 0:
        raise ValueError("No value exists")
    for val in dist_val:
        distance = pairwise_distance(val, ref)
        if distance >= dist_thres:
            return True
    return False

def embed_in_dist_percent_gt_ratio(value_list, ref, ratio, dist):
    if len(value_list) == 0: return False
    dist_list = sorted(dist_to_ref(value_list, ref))
    dist_at_ratio = dist_list[ceil(len(dist_list) * ratio) - 1]
    return dist_at_ratio <= dist

def embed_in_sorted_dist_percent_gt_ratio(sorted_dist_list, ratio, dist):
    if len(sorted_dist_list) == 0: return False
    dist_at_ratio = sorted_dist_list[ceil(len(sorted_dist_list) * ratio) - 1]
    return dist_at_ratio <= dist

def generate_pre_parallel_core(ns, start, end, queue):
    df = pd.DataFrame(ns.df[start : end], columns = ns.df_col, index = ns.df_idx[start : end])
    ratio_list = ns.ratio_list
    thres = ns.thres
    sample_size = ns.sample_size
    distance_list = sorted(ns.distance_list, reverse=True)
    df = df[df['dist_val'].apply(lambda x: len([v for v in x if not utils.contains_non_english_chars(v)]) >= 0.8 * len(x))]
    pre_list = []
    seen_ref_set = set()
    for i in range(sample_size):
        if i % 10 == 0: print(i)
        row = df.iloc[random.randint(0, len(df) - 1)]
        ref = random.choice(row['dist_val']).strip() 
        if ref in seen_ref_set:
            continue
        seen_ref_set.add(ref)
        if len(ref) < 2 or utils.contains_non_english_chars(ref) or is_oov(ref):
            continue
        seen_ref_set.add(ref)
        matching_rows = df
        matching_rows = df
        for ratio in ratio_list:
            for d in distance_list:
                matching_rows = matching_rows[matching_rows['dist_val'].apply(lambda x: embed_in_dist_percent_gt_ratio(x, ref, ratio, d))]
                if len(matching_rows) / len(df) < thres: break
                pre = ['embed', 1] + [ref] + [ratio, d]
                pre_list.append(tuple(pre))
    queue.put(pre_list)

def generate_pre_parallel(df, ratio_list, distance_list, thres, sample_size, n_proc):
    with mp.Manager() as manager:
        ns = manager.Namespace()
        ns.df = manager.list(df.values.tolist())
        ns.df_idx = manager.list(df.index.tolist())
        ns.df_col = df.columns
        ns.ratio_list = ratio_list
        ns.thres = thres
        ns.distance_list = distance_list
        ns.sample_size = sample_size
        with mp.Pool() as pool:
            start_list = [len(df) * i // n_proc for i in range(n_proc)]
            end_list = [len(df) * (i + 1) // n_proc for i in range(n_proc)]
            queue_list = [mp.Manager().Queue() for _ in range(n_proc)]
            pool.starmap(generate_pre_parallel_core, zip([ns] * n_proc, start_list, end_list, queue_list))
            results = []
            for q in queue_list:
                while not q.empty(): 
                    results.append(q.get())
            pre_list = []
            for r in results:
                for pre in r:
                    pre_list.append(pre)
            return list(set(pre_list))

def get_matching_rows(df, pre):
    assert pre[0] == 'embed'
    ref, ratio, dist = pre[2], pre[3], pre[4]
    df = df[df['dist_val'].apply(lambda x: len([v for v in x if not utils.contains_non_english_chars(v)]) >= 0.8 * len(x))]
    if len(df) == 0: return df
    df = df[df['dist_val'].apply(lambda x: mean(len(re.findall(r'[A-Za-z]+', v)) for v in x) <= 2)]
    if len(df) == 0: return df
    df = df[df['dist_val'].apply(lambda x: embed_in_dist_percent_gt_ratio(x, ref, ratio, dist))]
    return df

def get_matching_rows_parallel_core(ns, start, end, queue):
        df = pd.DataFrame(ns.df[start : end], columns = ns.df_col, index = ns.df_idx[start : end])
        pre_list = ns.pre_list
        idx_dict = {}
        for i in range(len(pre_list)):
            if i % 500 == 0: print(f'{i}/{len(pre_list)}')
            pre = pre_list[i]
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


def compute_cohenh(train: pd.DataFrame, matching_idx_dict: dict, pre: list, dist_thres: float):
    assert pre[0] == 'embed' 
    precond = utils.get_matching_rows_from_idx_dict(train, matching_idx_dict, pre)
    outdom = train.loc[~train.index.isin(precond.index)]
    ref = pre[2]
    indom_trigger = precond['dist_val'].apply(lambda x: max_dist_gt_thres(x, ref, dist_thres)).sum()
    outdom_trigger = outdom['dist_val'].apply(lambda x: max_dist_gt_thres(x, ref, dist_thres)).sum()
    indom_not_trigger = len(precond) - indom_trigger
    outdom_not_trigger = len(outdom) - outdom_trigger
    ch = utils.cohen_h(indom_trigger, indom_not_trigger, outdom_trigger, outdom_not_trigger)
    return ch, [indom_trigger, indom_not_trigger, outdom_trigger, outdom_not_trigger]

def compute_cohenh_parallel_core(ns, start, end, queue):
    df = pd.DataFrame(ns.df[start : end], columns = ns.df_col, index = ns.df_idx[start : end])
    pre_list = ns.pre_list
    thres_list = ns.thres_list
    matching_idx_dict = ns.matching_idx_dict
    for i in range(len(pre_list)):
        if i % 50 == 0: print(f"Progress: {i} / {len(pre_list)}")
        pre = pre_list[i]
        assert pre[0] == 'embed'
        precond = utils.get_matching_rows_from_idx_dict(df, matching_idx_dict, pre)
        outdom = df.loc[~df.index.isin(precond.index)]
        ref = pre[2]
        for thres in thres_list:
            indom_trigger = precond['dist_val'].apply(lambda x: max_dist_gt_thres(x, ref, thres)).sum()
            outdom_trigger = outdom['dist_val'].apply(lambda x: max_dist_gt_thres(x, ref, thres)).sum()
            indom_not_trigger = len(precond) - indom_trigger
            outdom_not_trigger = len(outdom) - outdom_trigger
            queue.put([tuple(pre) + ('embed', thres), [indom_trigger, indom_not_trigger, outdom_trigger, outdom_not_trigger]])
    
def compute_cohenh_parallel(df, matching_idx_dict, pre_list, thres_list, n_proc):
    with mp.Manager() as manager:
        ns = manager.Namespace()
        ns.df = manager.list(df.values.tolist())
        ns.df_idx = manager.list(df.index.tolist())
        ns.df_col = df.columns
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