import os,sys
import pickle
import warnings
from demo.demo_utils import demo_util
warnings.filterwarnings("ignore")

if not os.path.exists('./results/pr_curve'):
    # If it doesn't exist, create the directory
    os.makedirs('./results/pr_curve')

# name of the benchmark
benchmark_name = 'rt_bench'
train_corpus_name = 'rt_train'

# load SDC list
sdc_list_fname = f'./demo/{train_corpus_name}_sdc.pickle'
with open(sdc_list_fname, 'rb') as file:
    sdc_list = pickle.load(file)      
    
# apply SDC on the benchmark and obtain detected errors
benchmark = demo_util.load_corpus(benchmark_name)
detect_results = demo_util.apply_sdc(sdc_list, benchmark)

demo_util.process_results(detect_results, benchmark)
fig = demo_util.plot_pr_curves()
fig.savefig(f'./results/pr_curve/{train_corpus_name}_learnt_sdc_on_{benchmark_name}_prcurve.pdf')


# name of the benchmark
benchmark_name = 'st_bench'
train_corpus_name = 'rt_train'

# load SDC list
sdc_list_fname = f'./demo/{train_corpus_name}_sdc.pickle'
with open(sdc_list_fname, 'rb') as file:
    sdc_list = pickle.load(file)      
    
# apply SDC on the benchmark and obtain detected errors
benchmark = demo_util.load_corpus(benchmark_name)
detect_results = demo_util.apply_sdc(sdc_list, benchmark)

demo_util.process_results(detect_results, benchmark)
fig = demo_util.plot_pr_curves()
fig.savefig(f'./results/pr_curve/{train_corpus_name}_learnt_sdc_on_{benchmark_name}_prcurve.pdf')