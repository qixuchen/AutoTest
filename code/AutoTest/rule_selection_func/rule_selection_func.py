import os
import random
import pickle
import numpy as np
import pandas as pd
from config import config
from func import load_corpus
from functools import partial
from util import utils, sbert_utils, doduo_utils
from scipy.optimize import linprog
from check import embed_check, sherlock_check, doduo_check, pattern_check, sbert_check, pyfunc_check, validator_check

DFT_COHENH_THRES = 0.8
DFT_CONF_THRES = 0.1
DFT_NUM_RULE = 500
DFT_SIG_THRES = 0.08/100000
DFT_NUM_REPEAT = 1000
DFT_DELTA = 0.0001

def rule_utility_compute(validate, rule_corpus, rule_type_list, sbert_dist_val_embeddings_fname, doduo_intermediate_result_dir, doduo_dist_val_scores_fname):
    rule_fname_list = [os.path.join(config.dir.project_base_dir, config.dir.project_base.sdc_output, f'{rule_corpus}_{rule_type}.pickle') for rule_type in rule_type_list]
    rule_list = []
    for rule_fname in rule_fname_list:
        with open(rule_fname, 'rb') as file:
            rule_list += pickle.load(file)
    pre_list = list(set([r[0] for r in rule_list]))
    
    # load the preprocessed results for the validate corpus for sbert and doduo rule types
    # if no proprocessed results are available, compute them now 
    if any([rule[0][0] == 'sbert' for rule in rule_list]):
        if not os.path.exists(sbert_dist_val_embeddings_fname):
            print(f"SentenceBERT embedding file not found at {sbert_dist_val_embeddings_fname}, computing ...")
            sbert_dist_val_embeddings = sbert_utils.dist_val_embeddings_parallel(validate, n_proc = 8)
            with open(sbert_dist_val_embeddings_fname, 'wb') as file:
                pickle.dump(sbert_dist_val_embeddings, file)
                
        with open(os.path.join(config.dir.storage_root_dir, config.dir.storage_root.sbert, f'{load_corpus.CORPUS_NAME}_dist_val_embeddings.pkl'), 'rb') as file:
            sbert_dist_val_embeddings = pickle.load(file)
    else:
        sbert_dist_val_embeddings = None
    
    if any([rule[0][0] == 'doduo' for rule in rule_list]):
        if not os.path.exists(os.path.join(doduo_intermediate_result_dir, doduo_dist_val_scores_fname)):
            print(f"Doduo preprocessing result not found at {os.path.join(doduo_intermediate_result_dir, doduo_dist_val_scores_fname)}, computing ...")
            doduo_utils.dist_val_scores_parallel(validate, doduo_intermediate_result_dir, doduo_dist_val_scores_fname, n_proc = 15)
        doduo_dist_val_scores = pd.read_pickle(os.path.join(doduo_intermediate_result_dir, doduo_dist_val_scores_fname))
    else:
        doduo_dist_val_scores = None
        
    # compute the matching rows for the precondtion of each rule  
    test_matching_dict = utils.build_matching_idx_dict_from_pre_list_parallel(validate, pre_list, n_proc = 32, sbert_dist_val_embeddings = sbert_dist_val_embeddings, doduo_dist_val_scores = doduo_dist_val_scores)
    
    # get the outlier catched by each rule
    results = []
    if any([rule[1][0] == 'cta' for rule in rule_list]):
        sub_rule_list = [rule for rule in rule_list if rule[1][0] == 'cta']
        results += sherlock_check.sherlock_check_parallel(validate, test_matching_dict, sub_rule_list, n_proc = 48)
    if any([rule[1][0] == 'doduo' for rule in rule_list]):
        sub_rule_list = [rule for rule in rule_list if rule[1][0] == 'doduo']
        results += doduo_check.doduo_check_parallel(validate, test_matching_dict, sub_rule_list, n_proc = 15, doduo_dist_val_scores = doduo_dist_val_scores)
    if any([rule[1][0] == 'embed' for rule in rule_list]):
        sub_rule_list = [rule for rule in rule_list if rule[1][0] == 'embed']
        results += embed_check.embed_check_parallel(validate, test_matching_dict, sub_rule_list, n_proc = 48)
    if any([rule[1][0] == 'sbert' for rule in rule_list]):
        sub_rule_list = [rule for rule in rule_list if rule[1][0] == 'sbert']
        results += sbert_check.sbert_check_parallel(validate, test_matching_dict, sub_rule_list, n_proc = 8, sbert_dist_val_embeddings = sbert_dist_val_embeddings)
    if any([rule[1][0] == 'pattern' for rule in rule_list]):
        sub_rule_list = [rule for rule in rule_list if rule[1][0] == 'pattern']
        results += pattern_check.pattern_check(validate, test_matching_dict, sub_rule_list)
    if any([rule[1][0] == 'pyfunc' for rule in rule_list]):
        sub_rule_list = [rule for rule in rule_list if rule[1][0] == 'pyfunc']
        results += pyfunc_check.pyfunc_check(validate, test_matching_dict, sub_rule_list)
    if any([rule[1][0] == 'validator' for rule in rule_list]):
        sub_rule_list = [rule for rule in rule_list if rule[1][0] == 'validator']
        results += validator_check.validator_check(validate, test_matching_dict, sub_rule_list)
    return results


def randomized_rounding_rule_selection(rule_list, utility_dict, opt_x, num_rules):
        selected_rules = []
        for i in range(num_rules):
            if random.random() < opt_x[i]:
                selected_rules.append(rule_list[i])

        covered_elements = []
        selected_cost = 0
        for r in selected_rules:
            covered_elements += utility_dict[r][0]
            selected_cost += utility_dict[r][2]
        covered_elements = set(covered_elements)
        return selected_rules, covered_elements, selected_cost
    

def coarse_selection(rule_list, rule_outlier_results, params):
    cohenh_thres = params['cohenh_thres'] if 'cohenh_thres' in params else DFT_COHENH_THRES
    conf_thres = params['conf_thres'] if 'conf_thres' in params else DFT_CONF_THRES
    num_rule_thres = params['num_rule_thres'] if 'num_rule_thres' in params else DFT_NUM_RULE
    significance_thres = params['significance_thres'] if 'significance_thres' in params else DFT_SIG_THRES
    num_repeat = params['num_repeat_for_eval'] if 'num_repeat_for_eval' in params else DFT_NUM_REPEAT
    
    # compute the utility of each rule and store in a dict
    utility_dict = {}
    for r in rule_outlier_results:
        rule = r.iloc[0]['rule']
        rule[-1] = tuple(rule[-1])
        utility = []
        for idx, row in r.iterrows():
            if row['ground_truth'] == row['outlier']:
                utility.append(idx)
        if len(utility) > 0:
            cov = (rule[4][0] + rule[4][1])/sum(rule[4])
            cost = cov * rule[3]
            utility_dict[tuple(rule)] = [utility, cov, cost]
            
    rule_list = list(utility_dict.keys())
    rule_list = [r for r in rule_list if r[2] > cohenh_thres and r[3] < conf_thres and utils.chi2(*r[4])[1] < significance_thres]

    elements = []
    for rule in rule_list:
        elements += utility_dict[rule][0]
    elements = set(elements)
    element_list = list(elements)
    
    num_rules = len(rule_list)
    num_elements = len(element_list)
    cost_list = []
    for rule in rule_list:
        cost_list.append(utility_dict[rule][2])
        
    # objective: maximize \sum_{e} y_e
    obj_1 = np.zeros(num_rules)
    obj_2 = np.full(num_elements, -1)
    obj = np.concatenate((obj_1, obj_2))

    # size constraint: \sum_{S} x_S <= k
    size_cst_1 = np.ones(num_rules)
    size_cst_2 = np.zeros(num_elements)
    size_cst = np.concatenate((size_cst_1, size_cst_2))

    # cost constraint: \sum_{S} cost(S) * x_S <= C
    cost_cst_1 = np.array(cost_list)
    cost_cst_2 = np.zeros(num_elements)
    cost_cst = np.concatenate((cost_cst_1, cost_cst_2))

    # cover costraints: \sum_{e \in S} x_S <= y_e
    cover_cst = np.zeros([num_elements, num_rules + num_elements])
    for i in range(len(element_list)):
        e = element_list[i]
        cover_cst[i][num_rules + i] = 1
        for j in range(len(rule_list)):
            r = rule_list[j]
            if e in utility_dict[r][0]:
                cover_cst[i][j] = -1

    lhs_ineq = np.concatenate([np.array([size_cst, cost_cst]), cover_cst])
    rhs_ineq = np.concatenate([np.array([num_rule_thres, conf_thres]), np.zeros(num_elements)])
    bnd = (0, 1)
    
    opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, bounds=bnd)
    opt_x = opt.x
    
    # evaluate the performance of the selection
    randomize_rounding_selection = partial(randomized_rounding_rule_selection, rule_list, utility_dict, opt_x, num_rules)
    rule_list_covered_elements = []
    for r in rule_list:
        rule_list_covered_elements += utility_dict[r][0]
    rule_list_covered_elements = list(set(rule_list_covered_elements))
    size_res_list, cost_res_list, coverage_res_list = [], [], []
    
    for _ in range(num_repeat):
        coarse_selection_results = randomize_rounding_selection()
        size_res_list.append(len(coarse_selection_results[0]))
        cost_res_list.append(coarse_selection_results[2])
        coverage_res_list.append(len(coarse_selection_results[1]))

    print(f'avg coverage ratio: {sum(coverage_res_list) / (num_repeat * len(rule_list_covered_elements))}')
    print(f'avg FP cost ratio: {sum(cost_res_list) / (num_repeat * conf_thres)}')
    print(f'avg size ratio: {sum(size_res_list) / (num_repeat * num_rule_thres)}')
    return randomize_rounding_selection


def fine_selection(rule_list, rule_outlier_results, params):
    cohenh_thres = params['cohenh_thres'] if 'cohenh_thres' in params else DFT_COHENH_THRES
    conf_thres = params['conf_thres'] if 'conf_thres' in params else DFT_CONF_THRES
    num_rule_thres = params['num_rule_thres'] if 'num_rule_thres' in params else DFT_NUM_RULE
    significance_thres = params['significance_thres'] if 'significance_thres' in params else DFT_SIG_THRES
    delta = params['theta'] if 'theta' in params else DFT_DELTA
    num_repeat = params['num_repeat_for_eval'] if 'num_repeat_for_eval' in params else DFT_NUM_REPEAT
    
    # compute the utility of each rule and store in a dict
    # combined_results is used for the extra confidence requirement of fine-select
    combined_results = pd.DataFrame()
    for r in rule_outlier_results:
            for idx, row in r.iterrows():
                if row['ground_truth'] != row['outlier']: continue
                if idx not in combined_results.index:
                    combined_results = combined_results.append(row)
                else:
                    if row['conf'] < combined_results.loc[idx, 'conf']:
                        combined_results.loc[idx] = row
                        
    utility_dict = {}
    for r in rule_outlier_results:
        rule = r.iloc[0]['rule']
        rule[-1] = tuple(rule[-1])
        if tuple(rule) not in utility_dict.keys():
            utility = []
            for idx, row in r.iterrows():
                if row['ground_truth'] == row['outlier'] and row['conf'] <= combined_results.loc[idx]['conf'] + delta:
                    utility.append(idx)
            if len(utility) > 0:
                cov = (rule[4][0] + rule[4][1])/sum(rule[4])
                cost = cov * rule[3]
                utility_dict[tuple(rule)] = [utility, cov, cost]
                
    rule_list = list(utility_dict.keys())
    rule_list = [r for r in rule_list if r[2] > cohenh_thres and r[3] < conf_thres and utils.chi2(*r[4])[1] < significance_thres]
    
    elements = []
    for rule in rule_list:
        elements += utility_dict[rule][0]
    elements = set(elements)
    element_list = list(elements)
    
    num_rules = len(rule_list)
    num_elements = len(element_list)
    cost_list = []
    for rule in rule_list:
        cost_list.append(utility_dict[rule][2])
        
    # objective: maximize \sum_{e} y_e
    obj_1 = np.zeros(num_rules)
    obj_2 = np.full(num_elements, -1)
    obj = np.concatenate((obj_1, obj_2))

    # size constraint: \sum_{S} x_S <= k
    size_cst_1 = np.ones(num_rules)
    size_cst_2 = np.zeros(num_elements)
    size_cst = np.concatenate((size_cst_1, size_cst_2))

    # cost constraint: \sum_{S} cost(S) * x_S <= C
    cost_cst_1 = np.array(cost_list)
    cost_cst_2 = np.zeros(num_elements)
    cost_cst = np.concatenate((cost_cst_1, cost_cst_2))

    # cover costraints: \sum_{e \in S} x_S <= y_e
    cover_cst = np.zeros([num_elements, num_rules + num_elements])
    for i in range(len(element_list)):
        e = element_list[i]
        cover_cst[i][num_rules + i] = 1
        for j in range(len(rule_list)):
            r = rule_list[j]
            if e in utility_dict[r][0]:
                cover_cst[i][j] = -1

    lhs_ineq = np.concatenate([np.array([size_cst, cost_cst]), cover_cst])
    rhs_ineq = np.concatenate([np.array([num_rule_thres, conf_thres]), np.zeros(num_elements)])
    bnd = (0, 1)
    
    opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, bounds=bnd)
    opt_x = opt.x
    
    # evaluate the performance of the selection
    randomize_rounding_selection = partial(randomized_rounding_rule_selection, rule_list, utility_dict, opt_x, num_rules)
    rule_list_covered_elements = []
    for r in rule_list:
        rule_list_covered_elements += utility_dict[r][0]
    rule_list_covered_elements = list(set(rule_list_covered_elements))
    
    size_res_list, cost_res_list, coverage_res_list = [], [], []
    for _ in range(num_repeat):
        coarse_selection_results = randomize_rounding_selection()
        size_res_list.append(len(coarse_selection_results[0]))
        cost_res_list.append(coarse_selection_results[2])
        coverage_res_list.append(len(coarse_selection_results[1]))

    print(f'avg coverage ratio: {sum(coverage_res_list) / (num_repeat * len(rule_list_covered_elements))}')
    print(f'avg FP cost ratio: {sum(cost_res_list) / (num_repeat * conf_thres)}')
    print(f'avg size ratio: {sum(size_res_list) / (num_repeat * num_rule_thres)}')
    return randomize_rounding_selection