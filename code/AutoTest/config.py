from easydict import EasyDict as edict

config = edict()

config.dir = edict

# folder where AutoTest code base is stored. Need to be an absolute path, otherwise sherlock and doduo report errors
config.dir.project_base_dir = '/home/azureuser/AutoTest' 
config.dir.project_base = edict

# folder where trained SDCs are stored
config.dir.project_base.sdc_output = 'output/rule' 

# storage root for training corpus and intermediate results
config.dir.storage_root_dir = '/datadrive/qixu'
config.dir.storage_root = edict

# dirs for intermediate results under config.dir.storage_root_dir
config.dir.storage_root.coarse_select_lp = 'coarse_select_lp'
config.dir.storage_root.fine_select_lp = 'fine_select_lp'
config.dir.storage_root.coarse_select_rule = 'coarse_select_rule'
config.dir.storage_root.fine_select_rule = 'fine_select_rule'
config.dir.storage_root.validate_results = 'validate_results'
config.dir.storage_root.glove = 'GLoVe'
config.dir.storage_root.sbert = 'sbert_dist_embedding'
config.dir.storage_root.sherlock = 'preprocess/sherlock'
config.dir.storage_root.doduo = 'preprocess/doduo'

# folder under config.dir.storage_root_dir where training corpora are stored
config.dir.storage_root.train_corpora = 'train_corpora' 

# folder where ST_bench and RT_bench should be stored
config.dir.storage_root.benchmark = 'benchmark' 

# folder where the synthetic dataset for SDC selection should be stored
config.dir.storage_root.synth_dataset = 'synth_dataset' 

# folder for experimental results, can be ignored for release version
config.dir.storage_root.exp = 'experiment'


# # for docker

# # folder where AutoTest code base is stored. Need to be an absolute path, otherwise sherlock and doduo report errors
# config.dir.project_base_dir = '/root/AutoTest' 
# config.dir.project_base = edict

# # folder where trained SDCs are stored
# config.dir.project_base.sdc_output = 'output/rule' 

# # storage root for training corpus and intermediate results
# config.dir.storage_root_dir = '/root/storage'
# config.dir.storage_root = edict

# # dirs for intermediate results under config.dir.storage_root_dir
# config.dir.storage_root.coarse_select_lp = 'coarse_select_lp'
# config.dir.storage_root.fine_select_lp= 'fine_select_lp'
# config.dir.storage_root.coarse_select_rule = 'coarse_select_rule'
# config.dir.storage_root.fine_select_rule = 'fine_select_rule'
# config.dir.storage_root.validate_results = 'validate_results'
# config.dir.storage_root.glove = 'GLoVe'
# config.dir.storage_root.sbert = 'sbert_dist_embedding'
# config.dir.storage_root.sherlock = 'sherlock'
# config.dir.storage_root.doduo = 'doduo'

# # folder under config.dir.storage_root_dir where training corpora are stored
# config.dir.storage_root.train_corpora = 'train_corpora' 

# # folder where ST_bench and RT_bench should be stored
# config.dir.storage_root.benchmark = 'benchmark' 

# # folder where the synthetic dataset for SDC selection should be stored
# config.dir.storage_root.synth_dataset = 'synth_dataset' 

# # folder for experimental results, can be ignored for release version
# config.dir.storage_root.exp = 'experiment'