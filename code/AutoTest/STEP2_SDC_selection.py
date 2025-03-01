import warnings
warnings.filterwarnings("ignore")
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("corpus", choices=['rt_train', 'st_train', 'tablib', 'tablib_small'], help="Name of the corpus where SDC are learnt.")
args = parser.parse_args()

rule_corpus_dict = {'rt_train': 'PBICSV', 'st_train': 'Excel', 'tablib': 'Tablib_Sample_Large', 'tablib_small': 'Tablib_Small'}

rule_corpus = rule_corpus_dict[args.corpus]

import shutil
import os, pickle
import pandas as pd
from config import config
from func import load_corpus, convert_rule
from rule_selection_func import rule_selection_func

if not os.path.exists('./results/SDC'):
    # If it doesn't exist, create the directory
    os.makedirs('./results/SDC')

validate = load_corpus.load_corpus('PBICSV_validate')
rule_type_list = ['embed', 'sbert', 'cta', 'doduo', 'pattern', 'pyfunc', 'validator']

results_file_path = os.path.join(config.dir.storage_root_dir, config.dir.storage_root.validate_results, f'{load_corpus.CORPUS_NAME}_rule_{rule_corpus}_results.pickle')

if not os.path.exists(results_file_path):
    print('Cannot load precompute results, computing...')
    # these are for accelerating the computation of sbert and doduo rules
    # set them to where the intermediate results are stored (or where they will be stored if they are not available yet)
    sbert_dist_val_embeddings_fname = os.path.join(config.dir.storage_root_dir, config.dir.storage_root.sbert, f'{load_corpus.CORPUS_NAME}_dist_val_embeddings.pkl')
    doduo_intermediate_result_dir = os.path.join(config.dir.storage_root_dir, config.dir.storage_root.doduo)
    doduo_dist_val_scores_fname = f'{load_corpus.CORPUS_NAME}_dist_val_scores.pickle'

    rule_outlier_results = rule_selection_func.rule_utility_compute(validate, rule_corpus, rule_type_list, 
                                                                    sbert_dist_val_embeddings_fname, doduo_intermediate_result_dir, doduo_dist_val_scores_fname)

    print(f'Results will be stored in {results_file_path}.')
    with open(results_file_path, 'wb') as file:
        pickle.dump(rule_outlier_results, file)
    
else:    
    with open(results_file_path, 'rb') as file:
        rule_outlier_results = pickle.load(file) 
    print(f'Loaded precomputed results from {results_file_path}.')
    
    
rule_fname_list = [os.path.join(config.dir.project_base_dir, config.dir.project_base.sdc_output, f'{rule_corpus}_{rule_type}.pickle') for rule_type in rule_type_list]
rule_list = []
for rule_fname in rule_fname_list:
    with open(rule_fname, 'rb') as file:
        rule_list += pickle.load(file)
        
        
# fine_selection

fine_select_params = {
    'cohenh_thres': 0.8,
    'conf_thres': 0.1,
    'num_rule_thres': 500,
    'significance_thres': 0.08/100000,
    'delta': 0.0001,
    'num_repeat_for_eval': 1000,
}

randomize_rounding_selection = rule_selection_func.fine_selection(rule_list, rule_outlier_results, fine_select_params)
selected_rules = randomize_rounding_selection()[0]

fine_select_rule_file = os.path.join(config.dir.project_base_dir, config.dir.project_base.sdc_output, f'{rule_corpus}_fine_select.pickle')
with open(fine_select_rule_file, 'wb') as file:
    pickle.dump(selected_rules, file)
    
converted_rules = pd.DataFrame(columns=['type', 'pre-condition', 'post-condition', 'confidence', 'SDC'])
for rule in selected_rules:
    converted = convert_rule.convert_rule(rule)
    converted_rules = converted_rules.append(converted, ignore_index=True)   
converted_rules.sort_values('confidence', ascending = False).to_csv(f"./results/SDC/{args.corpus}_selected_sdc.csv", index = False, sep = '\t')