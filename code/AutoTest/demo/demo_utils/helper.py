
import pandas as pd
import numpy as np

def load_gt_and_predscore(fname):
    df = pd.read_csv(fname)
    return df['ground_truth'], df['pred_score']

def preprocess_precisions(precisions):
    for prec in precisions:
        prec[-1] = prec[-2]

    for prec in precisions:
        for i in range(len(prec) - 1):
            if prec[i] > prec[i + 1]:
                prec[i + 1]  = prec[i]
                
def pr_step(precision, recall, step_size = 0.001):
    precision, recall = precision[::-1], recall[::-1]
    steped_rec = np.arange(0, 1 + step_size, step_size)
    steped_prec = []
    last_index = 0 # for better efficiency
    for rec in steped_rec[:-1]:
        cur_index = last_index
        while recall[cur_index] <= rec: cur_index += 1
        r1, r2 = recall[cur_index - 1], recall[cur_index]
        p1, p2 = precision[cur_index - 1], precision[cur_index]
        prec = (rec - r1) / (r2 - r1) * (p2 - p1) + p1 
        steped_prec.append(prec)
        last_index = cur_index
    steped_prec.append(precision[-1])
    return steped_prec[::-1], steped_rec.tolist()[::-1]

def ave_prec_rec(precisions, recalls):
    return np.mean(np.array(precisions), axis=0), np.mean(np.array(recalls), axis=0)