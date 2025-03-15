import os
import pickle
import warnings
warnings.filterwarnings("ignore")
import argparse
from config import config
from func import load_corpus
from demo.demo_utils import demo_util


parser = argparse.ArgumentParser()
parser.add_argument("benchmark", choices=['rt_bench', 'st_bench'], help="Name of the benchmark.")
parser.add_argument("sdc", choices=['rt_train', 'st_train', 'tablib', 'tablib_small'], help="Name of the corpus where SDC are learnt.")
args = parser.parse_args()

benchmark_name_dict = {'rt_bench': 'labeled_benchmark', 'st_bench': 'excel_benchmark'}
rule_corpus_dict = {'rt_train': 'PBICSV', 'st_train': 'Excel', 'tablib': 'Tablib_Sample_Large', 'tablib_small': 'Tablib_Small'}

benchmark_name = benchmark_name_dict[args.benchmark]
benchmark = load_corpus.load_corpus(benchmark_name)

RULE_CORPUS = rule_corpus_dict[args.sdc]
rule_fname = os.path.join(config.dir.project_base_dir, config.dir.project_base.sdc_output, f'{RULE_CORPUS}_fine_select.pickle')
with open(rule_fname, 'rb') as file:
    rule_list = pickle.load(file) 

detect_results = demo_util.apply_sdc(rule_list, benchmark)

demo_util.process_results(detect_results, benchmark)
fig = demo_util.plot_pr_curves()
fig.savefig(f'./results/pr_curve/{args.sdc}_learnt_sdc_on_{args.benchmark}_prcurve.pdf')