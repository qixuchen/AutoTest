import os
from config import config

assert os.path.isdir(config.dir.storage_root_dir), 'Please provide a valid directory by setting the variable "config.dir.storage_root_dir" in "config.py" \
to store the intermediate result. Make sure the disk space for the directory is sufficiently large (several hunderds of GB at least).'

for dir_name in [config.dir.storage_root.coarse_select_lp, config.dir.storage_root.fine_select_lp, config.dir.storage_root.coarse_select_rule, config.dir.storage_root.fine_select_rule, config.dir.storage_root.validate_results, \
    config.dir.storage_root.glove, config.dir.storage_root.sbert, config.dir.storage_root.sherlock, config.dir.storage_root.doduo, config.dir.storage_root.train_corpora, config.dir.storage_root.benchmark, config.dir.storage_root.synth_dataset]:
        try:
            os.mkdir(os.path.join(config.dir.storage_root_dir, dir_name))
            print(f"Created folder '{dir_name}' under {config.dir.storage_root_dir}.")
        except FileExistsError:
            print(f"Directory '{os.path.join(config.dir.storage_root_dir, dir_name)}' already exists. Skipping ...")


assert os.path.isdir(config.dir.project_base_dir) and config.dir.project_base_dir == os.getcwd(), 'Set "config.dir.project_base_dir" to the folder where this file is located, and make sure using this folder as the working directory.'
try:
    os.mkdir(os.path.join(config.dir.project_base_dir, config.dir.project_base.sdc_output))
except FileExistsError:
    print(f"Directory '{os.path.join(config.dir.project_base_dir, config.dir.project_base.sdc_output)}' already exists. Skipping ...")

print('Configuration setup check completed.')