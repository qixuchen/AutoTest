import os
import csv
import pickle
from config import config
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

freq_dict_path = os.path.join(config.dir.storage_root_dir, 'PBICSV_Sampled/PBICSV_freq_dict_cell.filtered_num.freq_gt_2.txt') 
freq_dict = {}

with open(freq_dict_path, 'r') as file:
    reader = csv.reader(file, delimiter='\t')
    for row in reader:
        key = row[0]
        value = int(row[1])
        # a dirty fix for the bug caused by "russia" and "\"russia\""
        if key not in freq_dict.keys():
            freq_dict[key] = value

sbert_dict = {}
i = 0
for key in freq_dict.keys():
    i += 1
    sbert_dict[key] = model.encode(key)
    if i % 1000 == 0:
        print(i/len(freq_dict.keys()))

sbert_dict_path = os.path.join(config.dir.storage_root_dir, 'sBert_dict.pickle')
with open(sbert_dict_path, 'wb') as file:
    pickle.dump(sbert_dict, file)