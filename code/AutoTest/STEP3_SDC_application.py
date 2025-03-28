import warnings
warnings.filterwarnings("ignore")
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("benchmark", choices=['rt_bench', 'st_bench'], help="Name of the benchmark.")
parser.add_argument("sdc", choices=['rt_train', 'st_train', 'tablib', 'tablib_small'], help="Name of the corpus where SDC are learnt.")
args = parser.parse_args()

benchmark_name_dict = {'rt_bench': 'labeled_benchmark', 'st_bench': 'excel_benchmark'}
rule_corpus_dict = {'rt_train': 'PBICSV', 'st_train': 'Excel', 'tablib': 'Tablib_Sample_Large', 'tablib_small': 'Tablib_Small'}

benchmark_name = benchmark_name_dict[args.benchmark]
RULE_CORPUS = rule_corpus_dict[args.sdc]


import os
import pickle
import pandas as pd
from config import config
from func import load_corpus
from util import utils, sbert_utils, doduo_utils
from check import embed_check, sherlock_check, doduo_check, pattern_check, sbert_check, pyfunc_check, validator_check

if not os.path.exists('./results/detected_outliers'):
    # If it doesn't exist, create the directory
    os.makedirs('./results/detected_outliers')

benchmark = load_corpus.load_corpus(benchmark_name)

cohenh_thres = 0.8
conf_thres = 0.1
num_rule_thres = 500
sbert_dist_val_embeddings_fname = os.path.join(config.dir.storage_root_dir, config.dir.storage_root.sbert, f'{load_corpus.CORPUS_NAME}_dist_val_embeddings.pkl')
doduo_intermediate_result_dir = os.path.join(config.dir.storage_root_dir, config.dir.storage_root.doduo)
doduo_dist_val_scores_fname = os.path.join(config.dir.storage_root_dir, config.dir.storage_root.doduo, f'{load_corpus.CORPUS_NAME}_dist_val_scores.pickle')

rule_list_fname = os.path.join(config.dir.project_base_dir, config.dir.project_base.sdc_output, f'{RULE_CORPUS}_fine_select.pickle')


with open(rule_list_fname, 'rb') as file:
    rule_list = pickle.load(file)      
pre_list = list(set([r[0] for r in rule_list]))

sbert_dist_val_embeddings = None
doduo_dist_val_scores = None

if any([rule[0][0] == 'sbert' for rule in rule_list]):
    if not os.path.exists(sbert_dist_val_embeddings_fname):
        print("SentenceBERT embedding file not found, computing ...")
        sbert_dist_val_embeddings = sbert_utils.dist_val_embeddings_parallel(benchmark, n_proc = 8)
        with open(sbert_dist_val_embeddings_fname, 'wb') as file:
            pickle.dump(sbert_dist_val_embeddings, file)
            
    with open(sbert_dist_val_embeddings_fname, 'rb') as file:
        sbert_dist_val_embeddings = pickle.load(file)

if any([rule[0][0] == 'doduo' for rule in rule_list]):
    if not os.path.exists(doduo_dist_val_scores_fname):
        print("Doduo preprocessing result not found, computing ...")
        doduo_utils.dist_val_scores_parallel(benchmark, doduo_intermediate_result_dir, doduo_dist_val_scores_fname, n_proc = 15)
    doduo_dist_val_scores = pd.read_pickle(doduo_dist_val_scores_fname)
    
test_matching_dict = utils.build_matching_idx_dict_from_pre_list_parallel(benchmark, pre_list, n_proc = 32, sbert_dist_val_embeddings = sbert_dist_val_embeddings, doduo_dist_val_scores = doduo_dist_val_scores)


results = []
if any([rule[1][0] == 'cta' for rule in rule_list]):
    sub_rule_list = [rule for rule in rule_list if rule[1][0] == 'cta']
    results += sherlock_check.sherlock_check_parallel(benchmark, test_matching_dict, sub_rule_list, n_proc = 48)
if any([rule[1][0] == 'doduo' for rule in rule_list]):
    sub_rule_list = [rule for rule in rule_list if rule[1][0] == 'doduo']
    results += doduo_check.doduo_check_parallel(benchmark, test_matching_dict, sub_rule_list, n_proc = 15, doduo_dist_val_scores = doduo_dist_val_scores)
if any([rule[1][0] == 'embed' for rule in rule_list]):
    sub_rule_list = [rule for rule in rule_list if rule[1][0] == 'embed']
    results += embed_check.embed_check_parallel(benchmark, test_matching_dict, sub_rule_list, n_proc = 48)
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
if len(final_res) > 0:  
    final_res['conf'] = 1 - final_res['conf']

final_res = final_res.sort_values('conf', ascending = False).rename(columns={"rule": "SDC"})
final_res[['header', 'ground_truth', 'ground_truth_debatable', 'outlier', 'conf', 'SDC', 'dist_val']].to_csv(f"./results/detected_outliers/{args.sdc}_learnt_sdc_on_{args.benchmark}.csv", index=False)

# import warnings
# warnings.filterwarnings("ignore")
# import argparse
# import os
# import ast
# import pickle
# import pandas as pd
# from config import config
# from func import load_corpus
# from util import utils, sbert_utils, doduo_utils
# from check import embed_check, sherlock_check, doduo_check, pattern_check, sbert_check, pyfunc_check, validator_check


# parser = argparse.ArgumentParser()
# parser.add_argument("benchmark_fname", help="Path to the benchmark.")
# parser.add_argument("sdc_fname", help="Path to the file of learned SDCs.")
# args = parser.parse_args()

# benchmark_fname = os.path.splitext(os.path.basename(args.benchmark_fname))[0]
# sdc_fname = os.path.splitext(os.path.basename(args.sdc_fname))[0]

# if not os.path.exists('./results/detected_outliers'):
#     # If it doesn't exist, create the directory
#     os.makedirs('./results/detected_outliers')


# benchmark = pd.read_csv(args.benchmark_fname, sep = '\t')
# benchmark['dist_val'] = benchmark['dist_val'].apply(ast.literal_eval)

# sbert_dist_val_embeddings_fname = os.path.join(config.dir.storage_root_dir, config.dir.storage_root.sbert, f'{benchmark_fname}_dist_val_embeddings.pkl')
# doduo_intermediate_result_dir = os.path.join(config.dir.storage_root_dir, config.dir.storage_root.doduo)
# doduo_dist_val_scores_fname = os.path.join(config.dir.storage_root_dir, config.dir.storage_root.doduo, f'{benchmark_fname}_dist_val_scores.pickle')

# rule_df = pd.read_csv(args.sdc_fname, sep = '\t')
# rule_list = rule_df['SDC'].apply(ast.literal_eval).to_list()

# sbert_dist_val_embeddings = None
# doduo_dist_val_scores = None

# if any([rule[0][0] == 'sbert' for rule in rule_list]):
#     if not os.path.exists(sbert_dist_val_embeddings_fname):
#         print(f"SentenceBERT embedding file for {args.benchmark_fname} not found, computing ...")
#         sbert_dist_val_embeddings = sbert_utils.dist_val_embeddings_parallel(benchmark, n_proc = 8)
#         with open(sbert_dist_val_embeddings_fname, 'wb') as file:
#             pickle.dump(sbert_dist_val_embeddings, file)
            
#     with open(sbert_dist_val_embeddings_fname, 'rb') as file:
#         sbert_dist_val_embeddings = pickle.load(file)

# if any([rule[0][0] == 'doduo' for rule in rule_list]):
#     if not os.path.exists(doduo_dist_val_scores_fname):
#         print(f"Doduo preprocessing result for {args.benchmark_fname} not found, computing ...")
#         doduo_utils.dist_val_scores_parallel(benchmark, doduo_intermediate_result_dir, f'{benchmark_fname}_dist_val_scores.pickle', n_proc = 15)
#     doduo_dist_val_scores = pd.read_pickle(doduo_dist_val_scores_fname)

# pre_list = list(set([r[0] for r in rule_list]))
# test_matching_dict = utils.build_matching_idx_dict_from_pre_list_parallel(benchmark, pre_list, n_proc = 32, sbert_dist_val_embeddings = sbert_dist_val_embeddings, doduo_dist_val_scores = doduo_dist_val_scores)


# results = []
# if any([rule[1][0] == 'cta' for rule in rule_list]):
#     sub_rule_list = [rule for rule in rule_list if rule[1][0] == 'cta']
#     results += sherlock_check.sherlock_check_parallel(benchmark, test_matching_dict, sub_rule_list, n_proc = 48)
# if any([rule[1][0] == 'doduo' for rule in rule_list]):
#     sub_rule_list = [rule for rule in rule_list if rule[1][0] == 'doduo']
#     results += doduo_check.doduo_check_parallel(benchmark, test_matching_dict, sub_rule_list, n_proc = 15, doduo_dist_val_scores = doduo_dist_val_scores)
# if any([rule[1][0] == 'embed' for rule in rule_list]):
#     sub_rule_list = [rule for rule in rule_list if rule[1][0] == 'embed']
#     results += embed_check.embed_check_parallel(benchmark, test_matching_dict, sub_rule_list, n_proc = 48)
# if any([rule[1][0] == 'sbert' for rule in rule_list]):
#     sub_rule_list = [rule for rule in rule_list if rule[1][0] == 'sbert']
#     results += sbert_check.sbert_check_parallel(benchmark, test_matching_dict, sub_rule_list, n_proc = 8, sbert_dist_val_embeddings = sbert_dist_val_embeddings)
# if any([rule[1][0] == 'pattern' for rule in rule_list]):
#     sub_rule_list = [rule for rule in rule_list if rule[1][0] == 'pattern']
#     results += pattern_check.pattern_check(benchmark, test_matching_dict, sub_rule_list)
# if any([rule[1][0] == 'pyfunc' for rule in rule_list]):
#     sub_rule_list = [rule for rule in rule_list if rule[1][0] == 'pyfunc']
#     results += pyfunc_check.pyfunc_check(benchmark, test_matching_dict, sub_rule_list)
# if any([rule[1][0] == 'validator' for rule in rule_list]):
#     sub_rule_list = [rule for rule in rule_list if rule[1][0] == 'validator']
#     results += validator_check.validator_check(benchmark, test_matching_dict, sub_rule_list)
    
    
# final_res = pd.DataFrame()
# for r in results:
#     for idx, row in r.iterrows():
#         if idx not in final_res.index:
#             final_res = final_res.append(row)
#         else:
#             if row['conf'] < final_res.loc[idx, 'conf']:
#                 final_res.loc[idx] = row
# if len(final_res) > 0:  
#     final_res['conf'] = 1 - final_res['conf']

# final_res = final_res.sort_values('conf', ascending = False).rename(columns={"rule": "SDC"})
# print(final_res[['header', 'outlier', 'conf', 'dist_val', 'SDC']])
# final_res[['header', 'outlier', 'conf', 'dist_val', 'SDC']].to_csv(f"./results/detected_outliers/{sdc_fname}_on_{benchmark_fname}.csv", index=False, sep = '\t')