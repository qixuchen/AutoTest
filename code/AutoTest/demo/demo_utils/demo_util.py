import os
from config import config
import pickle
from func import load_corpus
import demo.demo_utils.helper as helper
import pandas as pd
import matplotlib.pyplot as plt
from util import utils, sbert_utils, doduo_utils
from exp_utils import exp_util
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from check import embed_check, sherlock_check, doduo_check, pattern_check, sbert_check, pyfunc_check, validator_check

CORPUS_NAME = 'rt_bench'


def apply_sdc(rule_list, benchmark):
    pre_list = list(set([r[0] for r in rule_list]))

    sbert_dist_val_embeddings_fname = os.path.join(config.dir.storage_root_dir, config.dir.storage_root.sbert, f'{load_corpus.CORPUS_NAME}_dist_val_embeddings.pkl')
    doduo_intermediate_result_dir = os.path.join(config.dir.storage_root_dir, config.dir.storage_root.doduo)
    doduo_dist_val_scores_fname = os.path.join(config.dir.storage_root_dir, config.dir.storage_root.doduo, f'{load_corpus.CORPUS_NAME}_dist_val_scores.pickle')
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
    return final_res


def process_results(final_res, benchmark):
    pred_outliers = pd.DataFrame({'outlier': final_res['outlier'], 'prob': final_res['conf']})
    ground_truth, pred_score = exp_util.get_gt_and_predscore(benchmark, pred_outliers)
    plot_data = pd.DataFrame({'ground_truth': ground_truth.astype(float), 'pred_score': pred_score.astype(float)})
    plot_data.to_csv(f'./demo/baseline_results/{CORPUS_NAME}/fine_select.csv', index = False)
    
    
def plot_pr_curves():
    fig, ax = plt.subplots(figsize = (12, 12))
    path = f"./demo/baseline_results/{CORPUS_NAME}"
    names = ['Glove', 'SentenceBERT', 'Sherlock', 'Doduo', 'Regex', 'Dataprep', 'Validators', 'AutoDetect', 'Katara', 'Vender-A', 'Vender-B', 'GPT-4', 'GPT-finetune', 'Fine_select']
    fnames = [os.path.join(path, x) for x in ['glove.csv', 'sbert.csv', 'cta.csv', 'doduo.csv', 'pattern.csv', 'dataprep.csv', 'validator.csv', 'autodetect.csv', 'kb.csv' , 'venderA.csv' , 'venderB.csv', \
                'gpt.csv', 'gpt_finetune.csv', 'fine_select.csv']]
    colors = ['navy', 'lightsalmon', 'red', 'turquoise', 'blueviolet', 'darkorange', 'tan', 'greenyellow', 'chocolate', 'steelblue', 'darkslategray', 'slategray', 'blue', 'green']
    markers = ['^', '2', 'x', 'o', '+', 's', 'p', 'h', '*', 'X', 'P', '1', '<', 'd']
    #precisions and recalls
    precisions, recalls = [], []

    # plot baseline
    x_tick = [0, 1]
    if CORPUS_NAME == 'rt_bench':
        y_tick = [25/1200, 25/1200] # 25 positive in 1200 cols
    else:
        y_tick = [22/1200, 22/1200] # 22 positive in 1200 cols
    ax.plot(x_tick, y_tick, label = "baseline", linestyle = 'dashed', linewidth=2, color = 'black')

    for fname in fnames:
        ground_truth, pred_score = helper.load_gt_and_predscore(fname)
        pre, rec, thres = precision_recall_curve(ground_truth, pred_score)
        helper.preprocess_precisions([pre])
        pre, rec = helper.pr_step(pre, rec)
        precisions.append(pre)
        recalls.append(rec)
        
    for i in range(len(names)):
        display = PrecisionRecallDisplay(
            recall = recalls[i],
            precision = precisions[i],
        )
        display.plot(ax = ax, marker = markers[i], markersize = 15, markeredgewidth = 2, markevery = 100, markerfacecolor='none', label = names[i], linestyle = 'solid', linewidth = 2.5, color = colors[i], drawstyle = 'default')

    ax.set_xlabel("Recall", fontsize = 30)
    ax.set_ylabel("Precision", fontsize = 30)
    ax.set_xlim([-0.02, 0.62])
    ax.set_ylim([-0.02, 1.02])
    ax.tick_params(axis = 'both', which = 'major', direction = "out", length = 5, width = 2.5, pad = 12, labelsize = 25)

    # set the legend and the axes
    handles, labels = display.ax_.get_legend_handles_labels()
    ax.legend(handles = handles, labels = labels, loc = "upper right", fontsize = 16)
    if CORPUS_NAME == 'rt_bench':
        ax.set_title("PR curves on Rt_bench", fontsize=30)
    else:
        ax.set_title("PR curves on St_bench", fontsize=30)
    
    plt.tight_layout()
    return fig