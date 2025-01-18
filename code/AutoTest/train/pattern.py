import random
import regex as re
from util import pattern_utils

DFT_RATIO_LIST = [0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99]
DFT_COV_THRES = 0.0003
DFT_SAMPLE_SIZE = 2000

def contain_digit(string):
    pattern = '\d'
    return bool(re.search(pattern, string))

def contain_special_char(string):
    pattern = '[@\-\+]'
    return bool(re.search(pattern, string))

def contain_alpha(string):
    pattern = '[a-zA-Z]'
    return bool(re.search(pattern, string))

def contains_non_ascii(string):
    for char in string:
        if ord(char) > 127:
            return True
    return False

def construct_possible_patterns(ref):
    pattern_list = []
    if contain_digit(ref):
        digit_match = ['wildcard']
    else:
        digit_match = ['exact']
    if contain_alpha(ref):
        alpha_match = ['wildcard']
    else:
        alpha_match = ['exact']
    for dm in digit_match:
        for am in alpha_match:
            if dm == 'exact' and am == 'exact': continue
            cur = ref
            pattern = "^"
            while len(cur)>0:
                if cur[0].isdigit() or cur[0] in ['-', '+']:
                    sub = re.search('[\+\-\d\.,]+', cur)
                    sub_length = len(sub.group())
                    if dm == 'exact':
                        pattern += str(sub.group())
                    elif dm == 'range':
                        pattern += '[\+\-\d\.,]{' + str(sub_length) + '}'
                    else:
                        pattern += '[\+\-\d\.,]+'
                    cur = cur[sub_length:]
                elif cur[0].isalpha():
                    sub = re.search('[a-zA-Z]+', cur)
                    sub_length = len(sub.group())
                    if am == 'exact':
                        pattern += str(sub.group())
                    elif am == 'range':
                        pattern += '[a-zA-Z]{' + str(sub_length) + '}'
                    else:
                        pattern += '[a-zA-Z]+'
                    cur = cur[sub_length:]
                else:
                    if cur[0] in ['\\', "\'", '\"', '\n', '\r', '|', '^', '$', '.', '*', '?', '(', ')', '[', ']', '{', '}']:
                        pattern += '\\' + cur[0]
                    else:
                        pattern += cur[0]
                    cur = cur[1:]
            pattern += '$'
            pattern_list.append(pattern)
    return pattern_list

# def generate_pre(train, matching_idx_dict, ratio, cov_thres, sample_size):
#     seen_pattern_set = set()
#     df = train.copy()
#     pre_list = []
#     for i in range(sample_size):
#         if i % 100 == 0:
#             print(f'{i}/{sample_size}')
#         row = df.iloc[random.randint(0, len(df) - 1)]
#         ref = row['dist_val'][random.randint(0, len(row['dist_val']) - 1)]
#         if contains_non_ascii(ref) or not contain_digit(ref):
#             continue
#         pattern_list = construct_possible_patterns(ref)
#         for pattern in pattern_list:
#             if pattern in seen_pattern_set or pattern_utils.pattern_matching_ratio(row['dist_val'], pattern) < ratio: 
#                 continue
#             seen_pattern_set.add(pattern)
#             matching_rows = df[df['dist_val'].apply(lambda x: pattern_utils.pattern_matching_ratio(x, pattern) >= ratio)]
#             coverage = len(matching_rows) / len(train)
#             if coverage < cov_thres: 
#                 continue
#             pre = ['pattern', 1, ref, pattern, ratio]
#             pre_list.append(pre)
#             matching_idx_dict[tuple(pre)] = matching_rows.index.to_list()
#             df = df.loc[~df.index.isin(matching_rows.index)]
#             if len(df) == 0: break
#     return pre_list

# def rule_generate(train, params):
    
#     ratio_list = params['ratio_list'] if 'ratio_list' in params else DFT_RATIO_LIST
#     coverage_thres = params['coverage_thres'] if 'coverage_thres' in params else DFT_COV_THRES
#     sample_size = params['sample_size'] if 'sample_size' in params else DFT_SAMPLE_SIZE
    
#     matching_idx_dict = {}
#     pre_list = []
#     print(f'Sampling {sample_size} values')
#     for ratio in ratio_list:
#         pre_list += generate_pre(train, matching_idx_dict, ratio, coverage_thres, sample_size)
    
#     print(f'Computing stats for {len(pre_list)} generated rules')  
#     rule_list = pattern_utils.compute_cohen_h(train, matching_idx_dict, pre_list)
#     return rule_list


def generate_pre(train, matching_idx_dict, ratio_list, cov_thres, sample_size):
    ratio_list = sorted(ratio_list)
    seen_pattern_set = set()
    pre_list = []
    for i in range(sample_size):
        if i % 20 == 0:
            print(f'{i}/{sample_size}')
            
        row = train.iloc[random.randint(0, len(train) - 1)]
        ref = row['dist_val'][random.randint(0, len(row['dist_val']) - 1)]
        if contains_non_ascii(ref) or not contain_digit(ref):
            continue
        
        pattern_list = construct_possible_patterns(ref)
        for pattern in pattern_list:
            if pattern in seen_pattern_set: 
                continue
            
            for ratio in ratio_list:
                matching_rows = train[train['dist_val'].apply(lambda x: pattern_utils.pattern_matching_ratio(x, pattern) >= ratio)]
                coverage = len(matching_rows) / len(train)
                if coverage < cov_thres: 
                    break
                
                pre = ['pattern', 1, ref, pattern, ratio]
                print(pre)
                pre_list.append(pre)
                matching_idx_dict[tuple(pre)] = matching_rows.index.to_list()
                
            seen_pattern_set.add(pattern)
            
    return pre_list

def rule_generate(train, params):
    
    ratio_list = params['ratio_list'] if 'ratio_list' in params else DFT_RATIO_LIST
    coverage_thres = params['coverage_thres'] if 'coverage_thres' in params else DFT_COV_THRES
    sample_size = params['sample_size'] if 'sample_size' in params else DFT_SAMPLE_SIZE
    
    matching_idx_dict = {}
    print(f'Sampling {sample_size} values')
    pre_list = generate_pre(train, matching_idx_dict, ratio_list, coverage_thres, sample_size)
    
    print(f'Computing stats for {len(pre_list)} generated rules')  
    rule_list = pattern_utils.compute_cohen_h(train, matching_idx_dict, pre_list)
    return rule_list