import os
import pickle
import pandas as pd
from config import config

def convert_rule(r):
    if r[0][0] == 'cta': 
        _, classname, ratio, s_in = r[0] 
        s_out = r[1][1]
        conf = 1 - r[3]
        domain = 'CTA'
        precond = f'{int(ratio * 100)} % column values have their Sherlock "{classname}"-classifier scores >= {s_in}'
        const = f'values whose Sherlock "{classname}"-classifier scores <= {s_out}'
        
    elif r[0][0] == 'doduo': 
        _, classname, ratio, s_in = r[0] 
        s_out = r[1][1]
        conf = 1 - r[3]
        domain = 'CTA'
        precond = f'{int(ratio * 100)} % column values have their Doduo "{classname}"-classifier scores >= {s_in}'
        const = f'values whose Doduo "{classname}"-classifier scores <= {s_out}"'
        
    elif r[0][0] == 'embed': 
        _, _, center, ratio, s_in = r[0] 
        s_out = r[1][1]
        conf = 1 - r[3]
        domain = 'Embedding'
        precond = f'{int(ratio * 100)} % column values have their Glove embedding distance to "{center}" < {s_in}'
        const = f'values whose Glove embedding distance to "{center}" > {s_out}'
        
    elif r[0][0] == 'sbert': 
        _, _, center, ratio, s_in = r[0] 
        s_out = r[1][1]
        conf = 1 - r[3]
        domain = 'Embedding'
        precond = f'{int(ratio * 100)} % column values have their Sentence-BERT embedding distance to "{center}" < {s_in}'
        const = f'values whose Sentence-BERT embedding distance to "{center}" > {s_out}'
        
    elif r[0][0] == 'pattern': 
        _, _, _, pattern, ratio = r[0] 
        conf = 1 - r[3]
        domain = 'Pattern'
        precond = f'{int(ratio * 100)} % of values match pattern "{pattern}"'
        const = f'values not matching pattern "{pattern}"'
        
    elif r[0][0] == 'pyfunc': 
        _, type, ratio = r[0] 
        conf = 1 - r[3]
        domain = 'Function'
        type_func_mapping = {
            'url': 'validate_url',
            'date': 'validate_date',
            'email': 'validate_email',
            'ip': 'validate_ip',
        }
        precond = f'{int(ratio * 100)} % of values return True on function "{type_func_mapping[type]}" in Dataprep'
        const = f'values that return False on function "{type_func_mapping[type]}" in Dataprep'
        
    elif r[0][0] == 'validator': 
        _, type, ratio = r[0] 
        conf = 1 - r[3]
        domain = 'Function'
        type_func_mapping = {
            'url': 'url',
            'email': 'email',
            'ip': 'ip_address',
        }
        precond = f'{int(ratio * 100)} % of values return True on function "{type_func_mapping[type]}" in Validators'
        const = f'values return False on function "{type_func_mapping[type]}" in Validators'
        
    else:
        raise Exception(f"Unrecognizable rule type: {r[0][0]}")
    
    return {'type': domain, 'pre-condition': precond, 'post-condition': const, 'confidence': conf, 'SDC': r}


if __name__=='__main__':
    rule_fname = os.path.join(config.dir.storage_root_dir, config.dir.storage_root.fine_select_rule, 'rule_PBICSV_cohen_h_0.8_wilson_0.1_num_rule_500/0.pickle')
    out_fname = "rules_readable.csv"
    
    with open(rule_fname, 'rb') as file:
        rules = pickle.load(file)
        
    converted_rules = pd.DataFrame(columns=['domain', 'precondition', 'constraint', 'confidence', 'SDC'])
    for rule in rules:
        converted = convert_rule(rule)
        converted_rules = converted_rules.append(converted, ignore_index=True)
        
    converted_rules.sort_values('confidence', ascending = False).to_csv(out_fname, index = False, sep = '\t')