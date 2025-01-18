import os
import pickle
import pandas as pd
import numpy as np
from config import config
from util import utils
from func import load_corpus
from check import embed_check, sherlock_check, doduo_check, pattern_check, sbert_check, pyfunc_check, validator_check

def compute_exp_stats(benchmark, outliers):
    FP, TP, debatable = 0, 0, 0
    for i in outliers.index:
        if outliers[i] == "N/A": continue
        elif outliers[i] in benchmark['ground_truth'][i]:
            TP += 1
        elif outliers[i] in benchmark['ground_truth_debatable'][i]:
            debatable += 1
        else:
            FP += 1
    return TP, debatable, FP


def eval_rule_list_on_benchmark(rule_list, benchmark):
    sbert_dist_val_embeddings = None
    doduo_dist_val_scores = None

    if any([rule[1][0] == 'sbert' for rule in rule_list]):    
        with open(os.path.join(config.dir.storage_root_dir, config.dir.storage_root.sbert, f'{load_corpus.CORPUS_NAME}_dist_val_embeddings.pkl'), 'rb') as file:
            sbert_dist_val_embeddings = pickle.load(file)

    if any([rule[1][0] == 'doduo' for rule in rule_list]):
        fname = os.path.join(config.dir.storage_root_dir, config.dir.storage_root.doduo, f'{load_corpus.CORPUS_NAME}_dist_val_scores.pickle')
        doduo_dist_val_scores = pd.read_pickle(fname)

    pre_list = list(set([tuple(rule[0]) for rule in rule_list]))
    test_matching_dict = utils.build_matching_idx_dict_from_pre_list_parallel(benchmark, pre_list, n_proc = 32, sbert_dist_val_embeddings = sbert_dist_val_embeddings, doduo_dist_val_scores = doduo_dist_val_scores)

    results = []
    if any([rule[1][0] == 'cta' for rule in rule_list]):
        sub_rule_list = [rule for rule in rule_list if rule[1][0] == 'cta']
        results += sherlock_check.sherlock_check_parallel(benchmark, test_matching_dict, sub_rule_list, n_proc = 32)
    if any([rule[1][0] == 'doduo' for rule in rule_list]):
        sub_rule_list = [rule for rule in rule_list if rule[1][0] == 'doduo']
        results += doduo_check.doduo_check_parallel(benchmark, test_matching_dict, sub_rule_list, n_proc = 15, doduo_dist_val_scores = doduo_dist_val_scores)
    if any([rule[1][0] == 'embed' for rule in rule_list]):
        sub_rule_list = [rule for rule in rule_list if rule[1][0] == 'embed']
        results += embed_check.embed_check_parallel(benchmark, test_matching_dict, sub_rule_list, n_proc = 32)
    if any([rule[1][0] == 'sbert' for rule in rule_list]):
        sub_rule_list = [rule for rule in rule_list if rule[1][0] == 'sbert']
        results += sbert_check.sbert_check_parallel(benchmark, test_matching_dict, sub_rule_list, n_proc = 8, sbert_dist_val_embeddings = sbert_dist_val_embeddings)
    if any([rule[1][0] == 'pattern' for rule in rule_list]):
        sub_rule_list = [rule for rule in rule_list if rule[1][0] == 'pattern']
        results += pattern_check.pattern_check(benchmark, test_matching_dict, sub_rule_list)
    if any([rule[1][0] == 'pyfunc' for rule in rule_list]):
        sub_rule_list = [rule for rule in rule_list if rule[1][0] == 'pyfunc']
        results += pyfunc_check.pyfunc_check(benchmark, test_matching_dict, sub_rule_list)
    if any([rule[1][0] == 'validator' for rule in rule_list]):
        sub_rule_list = [rule for rule in rule_list if rule[1][0] == 'validator']
        results += validator_check.validator_check(benchmark, test_matching_dict, sub_rule_list)
    
    final_res = pd.DataFrame()
    for r in results:
        for idx, row in r.iterrows():
            if idx not in final_res.index:
                final_res = final_res.append(row)
            else:
                if row['conf'] < final_res.loc[idx, 'conf']:
                    final_res.loc[idx] = row
    return final_res
                    
    # outliers = pd.Series(["N/A" for _ in range(len(benchmark))])
    # for i, r in final_res.iterrows():
    #     outliers[i] = r['outlier']

    # TP, debatable, FP = compute_exp_stats(benchmark, outliers)
    # print(f'TP: {TP}; debatable: {debatable}; FP: {FP}')


def get_gt_and_predscore(benchmark, pred_outliers):
    # pred_outliers should have two columns "outlier" and "prob"
    # where "outlier" is the predicted outlier and "prob" is the probability of being an outlier (i.e., 1 - conf)
    
    not_care = benchmark[(benchmark['ground_truth_debatable'].apply(lambda x: len(x) > 0)) & (benchmark['ground_truth'].apply(lambda x: len(x) == 0))].index.to_list()
    ground_truth = benchmark['ground_truth'].apply(lambda x: len(x) > 0).copy()
    pred_score = pd.Series([0.0] * len(benchmark))
    
    gt_FP = pd.Series(dtype = bool)
    pred_FP = pd.Series(dtype = float)

    for i, r in pred_outliers.iterrows():
        if  r['outlier'] in benchmark['ground_truth_debatable'][i] or r['outlier'] in benchmark['ground_truth'][i]: 
            pred_score.at[i] = r['prob']
        else:
            gt_FP = gt_FP.append(pd.Series([0]))
            pred_FP = pred_FP.append(pd.Series(r['prob']))
            
        
    ground_truth = ground_truth[~ground_truth.index.isin(not_care)]    
    pred_score = pred_score[~pred_score.index.isin(not_care)]  
    return pd.concat([ground_truth, gt_FP]), pd.concat([pred_score, pred_FP])


def get_gt_and_predscore_gpt_combine(benchmark, pred_outliers):
    # pred_outliers should have two columns "outlier" and "prob"
    # where "outlier" is the predicted outlier and "prob" is the probability of being an outlier (i.e., 1 - conf)
    
    not_care = benchmark[(benchmark['ground_truth_debatable'].apply(lambda x: len(x) > 0)) & (benchmark['ground_truth'].apply(lambda x: len(x) == 0))].index.to_list()
    ground_truth = benchmark['ground_truth'].apply(lambda x: len(x) > 0).copy()
    pred_score = pd.Series([0.0] * len(benchmark))
    
    gt_FP = pd.Series(dtype = bool)
    pred_FP = pd.Series(dtype = float)

    for i, r in pred_outliers.iterrows():
        if  r['outlier'] in benchmark['ground_truth_debatable'][i] or r['outlier'] in benchmark['ground_truth'][i]: 
            if pred_score.at[i] < r['prob']:
                pred_score.at[i] = r['prob']
        else:
            gt_FP = gt_FP.append(pd.Series([0]))
            pred_FP = pred_FP.append(pd.Series(r['prob']))
            
        
    ground_truth = ground_truth[~ground_truth.index.isin(not_care)]    
    pred_score = pred_score[~pred_score.index.isin(not_care)]  
    return pd.concat([ground_truth, gt_FP]), pd.concat([pred_score, pred_FP])

    # not_care = benchmark[benchmark['ground_truth_debatable'].apply(lambda x: len(x) > 0)].index.to_list()
    # ground_truth = benchmark['ground_truth'].apply(lambda x: len(x) > 0).copy()
    # pred_score = pd.Series([0] * len(benchmark))

    # for i, r in pred_outliers.iterrows():
    #     pred_score.at[i] = r['prob']
    #     if r['outlier'] in benchmark['ground_truth_debatable'][i] or r['outlier'] in benchmark['ground_truth'][i]: continue
    #     else:
    #         ground_truth.at[i] = 0
    #         if i in not_care:
    #             not_care.remove(i)

    # ground_truth = ground_truth[~ground_truth.index.isin(not_care)]    
    # pred_score = pred_score[~pred_score.index.isin(not_care)]    
    # return ground_truth, pred_score