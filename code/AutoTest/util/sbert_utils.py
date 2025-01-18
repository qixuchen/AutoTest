import os
import pickle
import random
import torch
import pandas as pd
import numpy as np
import multiprocessing as mp
from config import config
from collections import defaultdict
from scipy import spatial
from math import ceil
from sentence_transformers import SentenceTransformer
from util import utils

sbert_dict_path = os.path.join(config.dir.storage_root_dir, 'sBert_dict.pickle')
sbert_dict = {}
model = None
dict_count = 0
model_count = 0
tried_load_sbert_dict = False
torch.set_num_threads(6)

def load_sbert_dict():
    global sbert_dict
    with open(sbert_dict_path, 'rb') as file:
        sbert_dict = pickle.load(file)

def validate_model_and_sbert_dict():
    global sbert_dict, model, tried_load_sbert_dict
    if len(sbert_dict) == 0 and not tried_load_sbert_dict:
        try:
            tried_load_sbert_dict = True
            load_sbert_dict()
            print("sbert dict loaded")
        except:
            print("sbert dict not loaded")
    if model == None:
        model = SentenceTransformer('all-MiniLM-L6-v2')

def decide_embedding(val):
    validate_model_and_sbert_dict()
    global dict_count, model_count
    if val in sbert_dict:
        dict_count += 1
        return sbert_dict[val]
    else:
        model_count += 1
        return model.encode(val, show_progress_bar = False)
    
def pairwise_distance(val: str, ref: str) -> float:
    return spatial.distance.euclidean(decide_embedding(val), decide_embedding(ref))
    
def dist_val_embeddings(dist_val):
    return [decide_embedding(val) for val in dist_val]
    
def avg_embedding(dist_val):
    return np.mean(dist_val_embeddings(dist_val), axis = 0)

# def dist_to_ref(dist_val: list, ref: str) -> list:
#     if len(dist_val) == 0:
#         raise ValueError("No value exists")
#     validate_model_and_sbert_dict()
#     ref_embed = [decide_embedding(ref)]
#     val_embed = [decide_embedding(val) for val in dist_val]
#     distances = spatial.distance.cdist(val_embed, ref_embed)
#     mean_distances = np.mean(distances, axis = 1)
#     return mean_distances

# def embed_in_dist_percent_gt_ratio(value_list, ref, ratio, dist):
#     if len(value_list) == 0: return False
#     validate_model_and_sbert_dict()
#     dist_list = sorted(dist_to_ref(value_list, ref))
#     dist_at_ratio = dist_list[ceil(len(dist_list) * ratio) - 1]
#     return dist_at_ratio <= dist

def dist_embeddings_to_ref(dist_embeddings: list, ref_embed) -> list:
    if len(dist_embeddings) == 0:
        raise ValueError("No value exists")
    distances = spatial.distance.cdist(dist_embeddings, [ref_embed])
    mean_distances = np.mean(distances, axis = 1)
    return mean_distances

def embed_in_dist_percent_gt_ratio_with_dist_embeddings(dist_embeddings, ref_embed, ratio, dist):
    if len(dist_embeddings) == 0: return False
    dist_list = sorted(dist_embeddings_to_ref(dist_embeddings, ref_embed))
    dist_at_ratio = dist_list[ceil(len(dist_list) * ratio) - 1]
    return dist_at_ratio <= dist

def embed_in_sorted_dist_percent_gt_ratio(sorted_dist_list, ratio, dist):
    if len(sorted_dist_list) == 0: return False
    dist_at_ratio = sorted_dist_list[ceil(len(sorted_dist_list) * ratio) - 1]
    return dist_at_ratio <= dist

def get_matching_rows(df, pre, sbert_dist_val_embeddings, sbert_avg_embedding):
    assert pre[0] == 'sbert'
    ref, ratio, dist = pre[2], pre[3], pre[4]
    df = df[df['dist_val'].apply(lambda x: len([v for v in x if not utils.contains_non_english_chars(v)]) >= 0.8 * len(x))]
    if len(df) == 0: return df
    ref_embed = None
    for idx, row in df.iterrows(): # try accelerate by getting ref embedding from computed results
        if ref in row['dist_val']:
            ref_idx = row['dist_val'].index(ref)  
            ref_embed = sbert_dist_val_embeddings[idx][ref_idx]
            break
    if ref_embed is None:
        ref_embed = decide_embedding(ref)
    df = df[sbert_avg_embedding.apply(lambda x: spatial.distance.euclidean(x, ref_embed) <= dist)]
    if len(df) == 0: return df
    df = df[sbert_dist_val_embeddings.apply(lambda x: embed_in_dist_percent_gt_ratio_with_dist_embeddings(x, ref_embed, ratio, dist))]
    return df

def dist_val_embeddings_parallel_core(ns, start, end, queue):
    df = pd.DataFrame(ns.df[start : end], columns = ns.df_col, index = ns.df_idx[start : end])
    queue.put(df['dist_val'].apply(dist_val_embeddings))

def dist_val_embeddings_parallel(df, n_proc):
    with mp.Manager() as manager:
        ns = manager.Namespace()
        ns.df = manager.list(df.values.tolist())
        ns.df_idx = manager.list(df.index.tolist())
        ns.df_col = df.columns
        with mp.Pool() as pool:
            start_list = [len(df) * i // n_proc for i in range(n_proc)]
            end_list = [len(df) * (i + 1) // n_proc for i in range(n_proc)]
            queue_list = [mp.Manager().Queue() for _ in range(n_proc)]
            pool.starmap(dist_val_embeddings_parallel_core, zip([ns] * n_proc, start_list, end_list, queue_list))
            results = []
            for q in queue_list:
                while not q.empty(): 
                    results.append(q.get())
            dist_val_embeds = pd.Series()
            for r in results:
                dist_val_embeds = pd.concat([dist_val_embeds, r])
    return dist_val_embeds

def avg_embedding_parallel_core(ns, start, end, queue):
    df = pd.DataFrame(ns.df[start : end], columns = ns.df_col, index = ns.df_idx[start : end])
    queue.put(df['dist_val'].apply(avg_embedding))

def avg_embedding_parallel(df, n_proc):
    with mp.Manager() as manager:
        ns = manager.Namespace()
        ns.df = manager.list(df.values.tolist())
        ns.df_idx = manager.list(df.index.tolist())
        ns.df_col = df.columns
        with mp.Pool() as pool:
            start_list = [len(df) * i // n_proc for i in range(n_proc)]
            end_list = [len(df) * (i + 1) // n_proc for i in range(n_proc)]
            queue_list = [mp.Manager().Queue() for _ in range(n_proc)]
            pool.starmap(avg_embedding_parallel_core, zip([ns] * n_proc, start_list, end_list, queue_list))
            results = []
            for q in queue_list:
                while not q.empty(): 
                    results.append(q.get())
            avg_embed = pd.Series()
            for r in results:
                avg_embed = pd.concat([avg_embed, r])
    return avg_embed
        
def generate_pre_parallel_core(ns, start, end, queue):
    df = pd.DataFrame(ns.df[start : end], columns = ns.df_col, index = ns.df_idx[start : end])
    sbert_dist_val_embeddings = pd.Series(ns.sbert_dist_val_embeddings[start : end], index = ns.sbert_dist_val_embeddings_idx[start : end])
    sbert_avg_embedding = pd.Series(ns.sbert_avg_embedding[start : end], index = ns.sbert_avg_embedding_idx[start : end])
    ratio_list = ns.ratio_list
    thres = ns.thres
    sample_size = ns.sample_size
    distance_list = sorted(ns.distance_list, reverse=True)
    df = df[df['dist_val'].apply(lambda x: len([v for v in x if not utils.contains_non_english_chars(v)]) >= 0.8 * len(x))]
    pre_list = []
    seen_ref_set = set()
    for i in range(sample_size):
        if i % 2 == 0: print(i)
        row = df.iloc[random.randint(0, len(df) - 1)]
        ref = random.choice(row['dist_val']).strip() 
        if ref in seen_ref_set or len(ref) < 2:
            continue
        seen_ref_set.add(ref)
        ref_embed = decide_embedding(ref)
        matching_rows = df
        for ratio in ratio_list:
            for d in distance_list:
                matching_rows = matching_rows[sbert_avg_embedding.apply(lambda x: spatial.distance.euclidean(x, avg_embedding([ref])) <= d)]
                if len(matching_rows) / len(df) < thres: break
                matching_rows = matching_rows[sbert_dist_val_embeddings.apply(lambda x: embed_in_dist_percent_gt_ratio_with_dist_embeddings(x, ref_embed, ratio, d))]
                if len(matching_rows) / len(df) < thres: break
                pre = ['sbert', 1] + [ref] + [ratio, d]
                pre_list.append(tuple(pre))
    queue.put(pre_list)

def generate_pre_parallel(df, sbert_dist_val_embeddings, sbert_avg_embedding, ratio_list, distance_list, thres, sample_size, n_proc):
    with mp.Manager() as manager:
        ns = manager.Namespace()
        ns.df = manager.list(df.values.tolist())
        ns.df_idx = manager.list(df.index.tolist())
        ns.df_col = df.columns
        ns.sbert_dist_val_embeddings = manager.list(sbert_dist_val_embeddings.values.tolist())
        ns.sbert_dist_val_embeddings_idx = manager.list(sbert_dist_val_embeddings.index.tolist())
        ns.sbert_avg_embedding = manager.list(sbert_avg_embedding.values.tolist())
        ns.sbert_avg_embedding_idx = manager.list(sbert_avg_embedding.index.tolist())
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

def get_matching_rows_parallel_core(ns, start, end, queue):
    df = pd.DataFrame(ns.df[start : end], columns = ns.df_col, index = ns.df_idx[start : end])
    sbert_dist_val_embeddings = pd.Series(ns.sbert_dist_val_embeddings[start : end], index = ns.sbert_dist_val_embeddings_idx[start : end])
    sbert_avg_embedding = pd.Series(ns.sbert_avg_embedding[start : end], index = ns.sbert_avg_embedding_idx[start : end])
    pre_list = ns.pre_list
    idx_dict = {}
    for i in range(len(pre_list)):
        if i % 10 == 0: print(f'{i}/{len(pre_list)}')
        pre = pre_list[i]
        matching_rows = get_matching_rows(df, pre, sbert_dist_val_embeddings, sbert_avg_embedding)
        idx_dict[tuple(pre)] = matching_rows.index.to_list()
    queue.put(idx_dict)

def get_matching_rows_parallel(df, pre_list, sbert_dist_val_embeddings, sbert_avg_embedding, n_proc):
    with mp.Manager() as manager:
        ns = manager.Namespace()
        ns.df = manager.list(df.values.tolist())
        ns.df_idx = manager.list(df.index.tolist())
        ns.df_col = df.columns
        ns.sbert_dist_val_embeddings = manager.list(sbert_dist_val_embeddings.values.tolist())
        ns.sbert_dist_val_embeddings_idx = manager.list(sbert_dist_val_embeddings.index.tolist())
        ns.sbert_avg_embedding = manager.list(sbert_avg_embedding.values.tolist())
        ns.sbert_avg_embedding_idx = manager.list(sbert_avg_embedding.index.tolist())
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

def max_dist_gt_thres(dist_val: list, ref: str, dist_thres) -> list:
    if len(dist_val) == 0:
        raise ValueError("No value exists")
    for val in dist_val:
        distance = pairwise_distance(val, ref)
        if distance >= dist_thres:
            return True
    return False

def max_dist_gt_thres_with_dist_embeddings(dist_embeddings: list, ref_embed, dist_thres) -> list:
    if len(dist_embeddings) == 0:
        raise ValueError("No value exists")
    for val_embed in dist_embeddings:
        distance = spatial.distance.euclidean(val_embed, ref_embed)
        if distance >= dist_thres:
            return True
    return False

def compute_cohenh_parallel_core(ns, start, end, queue):
    df = pd.DataFrame(ns.df[start : end], columns = ns.df_col, index = ns.df_idx[start : end])
    sbert_dist_val_embeddings = pd.Series(ns.sbert_dist_val_embeddings[start : end], index = ns.sbert_dist_val_embeddings_idx[start : end])
    pre_list = ns.pre_list
    thres_list = ns.thres_list
    matching_idx_dict = ns.matching_idx_dict
    for i in range(len(pre_list)):
        if i % 10 == 0: print(f"Progress: {i} / {len(pre_list)}")
        pre = pre_list[i]
        assert pre[0] == 'sbert'
        precond = utils.get_matching_rows_from_idx_dict(df, matching_idx_dict, pre)
        outdom = df.loc[~df.index.isin(precond.index)]
        precond_dist_embeddings = sbert_dist_val_embeddings.loc[sbert_dist_val_embeddings.index.isin(precond.index)]
        outdom_dist_embeddings = sbert_dist_val_embeddings.loc[sbert_dist_val_embeddings.index.isin(outdom.index)]
        ref = pre[2]
        ref_embed = decide_embedding(ref)
        for thres in thres_list:
            indom_trigger = precond_dist_embeddings.apply(lambda x: max_dist_gt_thres_with_dist_embeddings(x, ref_embed, thres)).sum()
            outdom_trigger = outdom_dist_embeddings.apply(lambda x: max_dist_gt_thres_with_dist_embeddings(x, ref_embed, thres)).sum()
            indom_not_trigger = len(precond) - indom_trigger
            outdom_not_trigger = len(outdom) - outdom_trigger
            queue.put([tuple(pre) + ('sbert', thres), [indom_trigger, indom_not_trigger, outdom_trigger, outdom_not_trigger]])
    
def compute_cohenh_parallel(df, sbert_dist_val_embeddings, matching_idx_dict, pre_list, thres_list, n_proc):
    with mp.Manager() as manager:
        ns = manager.Namespace()
        ns.df = manager.list(df.values.tolist())
        ns.df_idx = manager.list(df.index.tolist())
        ns.df_col = df.columns
        ns.pre_list = manager.list(pre_list)
        ns.thres_list = manager.list(thres_list)
        ns.matching_idx_dict = manager.dict(matching_idx_dict)
        ns.sbert_dist_val_embeddings = manager.list(sbert_dist_val_embeddings.values.tolist())
        ns.sbert_dist_val_embeddings_idx = manager.list(sbert_dist_val_embeddings.index.tolist())
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