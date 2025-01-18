import warnings
warnings.filterwarnings("ignore")
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("corpus", choices=['rt_train', 'st_train', 'tablib', 'tablib_small'], help="Name of the corpus where SDC are learnt.")
args = parser.parse_args()

rule_corpus_dict = {'rt_train': 'PBICSV', 'st_train': 'Excel', 'tablib': 'Tablib_Sample_Large', 'tablib_small': 'Tablib_Small'}

train_corpus_name = rule_corpus_dict[args.corpus]

import os, pickle
from config import config
from func import load_corpus
from train import glove, sbert, sherlock, doduo, pattern, pyfunc, validator

train = load_corpus.load_corpus(train_corpus_name)
print(f"training set size: {len(train)}")

rule_result_dir = os.path.join(config.dir.project_base_dir, config.dir.project_base.sdc_output)
WRITE_RULE = True

def write_rule_to_file(rule_list, rule_result_dir, rule_type):
    rule_fname = os.path.join(rule_result_dir, f'{load_corpus.CORPUS_NAME}_{rule_type}.pickle')
    with open(rule_fname, 'wb') as file:
        pickle.dump(rule_list, file)
        
# coverage_thres = 0.003
coverage_thres = 0.0003


# glove rules

glove_params = {
    'coverage_thres': coverage_thres,
    'sample_size': 25,
    'ratio_list': [0.8],
    'lb_list': [3 + 0.5 * i for i in range(9)],
    'ub_list': [5 + 0.5 * i for i in range(15)],
    'n_proc': 48
}

rule_list = glove.rule_generate(train, glove_params)
if WRITE_RULE:
    write_rule_to_file(rule_list, rule_result_dir, 'embed')
    

# sbert rules

sbert_params = {
    'coverage_thres': coverage_thres,
    'sample_size': 40,
    'ratio_list': [0.8],
    'lb_list': [0.8 + 0.1 * i for i in range(8)],
    'ub_list': [1.1 + 0.025 * i for i in range(17)],
    'sbert_dist_val_embeddings_fname': os.path.join(config.dir.storage_root_dir, config.dir.storage_root.sbert, f'{load_corpus.CORPUS_NAME}_dist_val_embeddings.pkl'),
    'n_proc': 6
}

rule_list = sbert.rule_generate(train, sbert_params)
if WRITE_RULE:
    write_rule_to_file(rule_list, rule_result_dir, 'sbert')
    
    
# sherlock rules

sherlock_params = {
    'coverage_thres': coverage_thres,
    'ratio_list': [0.8, 0.9, 0.95],
    'score_bars': [0.1 + 0.05 * i for i in range(15)],
    'thres_list': [0, 0.002, 0.005, 0.007, 0.01, 0.02, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3],
    'intermediate_result_dir': os.path.join(config.dir.storage_root_dir, config.dir.storage_root.sherlock),
    'min_label_score_fname': f'{load_corpus.CORPUS_NAME}_min_scores_cta.pickle',
    'n_proc': 32
}

rule_list = sherlock.rule_generate(train, sherlock_params)
if WRITE_RULE:
    write_rule_to_file(rule_list, rule_result_dir, 'cta')
    
    
# doduo rules

doduo_params = {
    'coverage_thres': coverage_thres,
    'ratio_list': [0.8],
    'score_bars': [4, 4.5],
    'thres_list': [-2.5, -2, -1.5, -1],
    'intermediate_result_dir': os.path.join(config.dir.storage_root_dir, config.dir.storage_root.doduo),
    'dist_val_scores_fname': f'{load_corpus.CORPUS_NAME}_dist_val_scores.pickle',
    'n_proc': 15
}

rule_list = doduo.rule_generate(train, doduo_params)
if WRITE_RULE:
    write_rule_to_file(rule_list, rule_result_dir, 'doduo')
    

# pattern rules

pattern_params = {
    'ratio_list': [0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99],
    'coverage_thres': coverage_thres,
    'sample_size': 5000
}

rule_list = pattern.rule_generate(train, pattern_params)
if WRITE_RULE:
    write_rule_to_file(rule_list, rule_result_dir, 'pattern')
    
    
# pyfunc rules

pyfunc_params = {
    'type_list': ['url', 'ip', 'date', 'email'],
    'ratio_list': [0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99]
}

rule_list = pyfunc.rule_generate(train, pyfunc_params)
if WRITE_RULE:
    write_rule_to_file(rule_list, rule_result_dir, 'pyfunc')
    
    
# validator rules

validator_params = {
    'type_list': ['ip', 'email', 'url'],
    'ratio_list': [0.8, 0.9],
    'n_proc': 32
}

rule_list = validator.rule_generate(train, validator_params)
if WRITE_RULE:
    write_rule_to_file(rule_list, rule_result_dir, 'validator')