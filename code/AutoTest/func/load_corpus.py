import os
import pandas as pd
from config import config


CORPUS_NAME = 'PBICSV'

def load_corpus(corpus_name):
    corpus_loading_fn = {
        'PBICSV': load_PBICSV,
        'Excel': load_ExcelCtrT,
        'WebTable': load_WebTable,
        'WebTable_Small': load_WebTable_Small,
        'WebTable_Large': load_WebTable_Large,
        'WebTable_Cat': load_WebTable_Cat,
        'Tablib_Sample': load_Tablib_Sample,
        'Tablib_Small': load_Tablib_Small,
        'Tablib_Sample_Large': load_Tablib_Sample_Large,
        'PBICSV_validate': load_PBICSV_validate,
        'labeled_benchmark': load_labeled_benchmark,
        'excel_benchmark': load_excel_benchmark,
        'data_cleaning_benchmark': load_data_cleaning_benchmark,
        'labeled_benchmark_5percent': load_labeled_benchmark_5percent,
        'labeled_benchmark_10percent': load_labeled_benchmark_10percent,
        'labeled_benchmark_20percent': load_labeled_benchmark_20percent,
        'excel_benchmark_5percent': load_excel_benchmark_5percent,
        'excel_benchmark_10percent': load_excel_benchmark_10percent,
        'excel_benchmark_20percent': load_excel_benchmark_20percent,
        'len500': load_len500
    }
    assert corpus_name in corpus_loading_fn, 'Invalid corpus name.'
    return corpus_loading_fn[corpus_name]()

def load_PBICSV():
    global CORPUS_NAME
    csv_file = os.path.join(config.dir.storage_root_dir, config.dir.storage_root.train_corpora, 'RT_train.txt')
    schema = {
        'fname': str,
        'col_header': str,
        'dist_val_str' : str,
        'dist_val_count' : int
    }

    corpus = pd.read_csv(csv_file, dtype = schema, sep = '\t', error_bad_lines = False)
    corpus['dist_val'] = corpus['dist_val_str'].apply(lambda x: str(x).split("___"))
    corpus['dist_val'] = corpus['dist_val'].apply(lambda x: [] if not type(x) == list else x)
    del corpus['dist_val_str']
    CORPUS_NAME = 'PBICSV'
    return corpus

def load_WebTable():
    global CORPUS_NAME
    csv_file = os.path.join(config.dir.storage_root_dir, 'WebTableAll/WebTable_sampled_500000.txt')
    header = ['val_str', 'dist_val_str', 'dist_val_count']
    schema = {
        'val_str' : str,
        'dist_val_str' : str,
        'dist_val_count' : int
    }

    corpus = pd.read_csv(csv_file, dtype = schema, header = None, names = header, sep = '\t', usecols = ['val_str', 'dist_val_str', 'dist_val_count'], error_bad_lines = False)
    corpus = corpus.drop_duplicates(subset=['val_str']).reset_index(drop=True)
    corpus['dist_val'] = list(corpus['dist_val_str'].str.split("___"))
    corpus['dist_val'] = corpus['dist_val'].apply(lambda x: [] if not type(x) == list else x)
    del corpus['val_str']
    del corpus['dist_val_str']
    CORPUS_NAME = 'WebTable'
    return corpus

def load_WebTable_Cat():
    global CORPUS_NAME
    csv_file = os.path.join(config.dir.storage_root_dir, 'WebTableAll/WebTable_Cat_sampled_400000.txt')
    header = ['val_str', 'dist_val_str', 'dist_val_count']
    schema = {
        'val_str' : str,
        'dist_val_str' : str,
        'dist_val_count' : int
    }

    corpus = pd.read_csv(csv_file, dtype = schema, header = None, names = header, sep = '\t', usecols = ['val_str', 'dist_val_str', 'dist_val_count'], error_bad_lines = False)
    corpus = corpus.drop_duplicates(subset=['val_str']).reset_index(drop=True)
    corpus['dist_val'] = list(corpus['dist_val_str'].str.split("___"))
    corpus['dist_val'] = corpus['dist_val'].apply(lambda x: [] if not type(x) == list else x)
    corpus['dist_val'] = corpus['dist_val'].apply(lambda x: [str(v) for v in x if str(v).lower() != 'nan' and str(v).lower() != ''])
    corpus['dist_val_count'] = corpus['dist_val'].apply(lambda x: len(x))
    corpus['avg_val_length'] = corpus['dist_val'].apply(lambda x: sum([len(v) for v in x])/len(x))
    corpus = corpus[corpus['avg_val_length'] < 60].reset_index(drop=True)
    del corpus['val_str']
    del corpus['dist_val_str']
    CORPUS_NAME = 'WebTable_Cat'
    return corpus

def load_WebTable_Small():
    global CORPUS_NAME
    csv_file = os.path.join(config.dir.storage_root_dir, 'WebTableAll/WebTable_Cat_sampled_300000.txt')
    header = ['val_str', 'dist_val_str', 'dist_val_count']
    schema = {
        'val_str' : str,
        'dist_val_str' : str,
        'dist_val_count' : int
    }

    corpus = pd.read_csv(csv_file, dtype = schema, header = None, names = header, sep = '\t', usecols = ['val_str', 'dist_val_str', 'dist_val_count'], error_bad_lines = False)
    corpus = corpus.drop_duplicates(subset=['val_str']).reset_index(drop=True)
    corpus['dist_val'] = list(corpus['dist_val_str'].str.split("___"))
    corpus['dist_val'] = corpus['dist_val'].apply(lambda x: [] if not type(x) == list else x)
    corpus['dist_val'] = corpus['dist_val'].apply(lambda x: [str(v) for v in x if str(v).lower() != 'nan' and str(v).lower() != ''])
    corpus['dist_val_count'] = corpus['dist_val'].apply(lambda x: len(x))
    corpus = corpus[(corpus['dist_val_count'] >= 5) & (corpus['dist_val_count'] <= 1000)].reset_index(drop=True)
    corpus['avg_val_length'] = corpus['dist_val'].apply(lambda x: sum([len(v) for v in x])/len(x))
    corpus = corpus[corpus['avg_val_length'] < 60].reset_index(drop=True)
    del corpus['val_str']
    del corpus['dist_val_str']
    CORPUS_NAME = 'WebTable_Small'
    return corpus[:int(len(corpus) * 0.2)]

def load_WebTable_Large():
    global CORPUS_NAME
    csv_file = os.path.join(config.dir.storage_root_dir, 'WebTableAll/WebTable_sampled_800000.txt')
    header = ['val_str', 'dist_val_str', 'dist_val_count']
    schema = {
        'val_str' : str,
        'dist_val_str' : str,
        'dist_val_count' : int
    }

    corpus = pd.read_csv(csv_file, dtype = schema, header = None, names = header, sep = '\t', usecols = ['val_str', 'dist_val_str', 'dist_val_count'], error_bad_lines = False)
    corpus = corpus.drop_duplicates(subset=['val_str']).reset_index(drop=True)
    corpus['dist_val'] = list(corpus['dist_val_str'].str.split("___"))
    corpus['dist_val'] = corpus['dist_val'].apply(lambda x: [] if not type(x) == list else x)
    del corpus['val_str']
    del corpus['dist_val_str']
    CORPUS_NAME = 'WebTable_Large'
    return corpus

def load_Tablib_Sample():
    global CORPUS_NAME
    csv_file = os.path.join(config.dir.storage_root_dir, config.dir.storage_root.train_corpora, 'Tablib_All.txt')
    header = ['val_str', 'dist_val_str', 'dist_val_count']
    schema = {
        'val_str' : str,
        'dist_val_str' : str,
        'dist_val_count' : int
    }

    corpus = pd.read_csv(csv_file, dtype = schema, header = None, names = header, sep = '\t', usecols = ['val_str', 'dist_val_str', 'dist_val_count'], error_bad_lines = False)
    corpus = corpus.drop_duplicates(subset=['val_str']).reset_index(drop=True)
    corpus['dist_val'] = list(corpus['dist_val_str'].str.split("___"))
    corpus['dist_val'] = corpus['dist_val'].apply(lambda x: [] if not type(x) == list else x)
    corpus['dist_val'] = corpus['dist_val'].apply(lambda x: [str(v) for v in x if str(v).lower() != 'nan' and str(v).lower() != ''])
    corpus['dist_val_count'] = corpus['dist_val'].apply(lambda x: len(x))
    corpus['avg_val_length'] = corpus['dist_val'].apply(lambda x: sum([len(v) for v in x])/len(x))
    corpus = corpus[corpus['avg_val_length'] < 60].reset_index(drop=True)
    del corpus['val_str']
    del corpus['dist_val_str']
    CORPUS_NAME = 'Tablib_Sample'
    return corpus

def load_Tablib_Small():
    global CORPUS_NAME
    csv_file = os.path.join(config.dir.storage_root_dir, config.dir.storage_root.train_corpora, 'Tablib_Small.txt')
    header = ['dist_val_str', 'dist_val_count']
    schema = {
        'dist_val_str' : str,
        'dist_val_count' : int
    }

    corpus = pd.read_csv(csv_file, dtype = schema, sep = '\t', usecols = ['dist_val_str', 'dist_val_count'], error_bad_lines = False)
    corpus['dist_val'] = list(corpus['dist_val_str'].str.split("___"))
    corpus['dist_val'] = corpus['dist_val'].apply(lambda x: [] if not type(x) == list else x)
    del corpus['dist_val_str']
    CORPUS_NAME = 'Tablib_Small'
    
    return corpus

def load_Tablib_Sample_Large():
    global CORPUS_NAME
    csv_file = os.path.join(config.dir.storage_root_dir, config.dir.storage_root.train_corpora, 'Tablib.txt')
    schema = {
        'dist_val_str' : str,
        'dist_val_count' : int
    }

    corpus = pd.read_csv(csv_file, dtype = schema, sep = '\t', error_bad_lines = False)
    corpus['dist_val'] = corpus['dist_val_str'].apply(lambda x: str(x).split("___"))
    corpus['dist_val'] = corpus['dist_val'].apply(lambda x: [] if not type(x) == list else x)
    del corpus['dist_val_str']
    CORPUS_NAME = 'Tablib_Sample_Large'
    return corpus

def load_ExcelCtrT():
    global CORPUS_NAME
    csv_file = os.path.join(config.dir.storage_root_dir, config.dir.storage_root.train_corpora, 'ST_train.txt')
    schema = {
        'fname1': str,
        'col_header': str,
        'dist_val_str' : str,
        'dist_val_count' : int
    }

    corpus = pd.read_csv(csv_file, dtype = schema, sep = '\t', error_bad_lines = False)
    corpus['dist_val'] = corpus['dist_val_str'].apply(lambda x: str(x).split("___"))
    corpus['dist_val'] = corpus['dist_val'].apply(lambda x: [] if not type(x) == list else x)
    del corpus['dist_val_str']
    CORPUS_NAME = 'Excel'
    return corpus

def load_PBICSV_validate():
    global CORPUS_NAME
    file = os.path.join(config.dir.storage_root_dir, config.dir.storage_root.synth_dataset, 'synthetic.txt')
    header = ['col_header', 'ground_truth', 'dist_val_count', 'dist_val_str']
    schema = {
        'col_header': str,
        'ground_truth': str,
        'dist_val_count' : int,
        'dist_val_str' : str
    }
    df = pd.read_csv(file, dtype = schema, header = None, names = header, sep = '\t', usecols = ['col_header', 'ground_truth', 'dist_val_str', 'dist_val_count'], on_bad_lines ='skip')
    df['dist_val'] = df['dist_val_str'].apply(lambda x: str(x).split("___"))
    CORPUS_NAME = 'PBICSV_validate'
    return df.reset_index()

def load_labeled_benchmark():
    global CORPUS_NAME
    def eval_string(s):
        if type(s) == str:
            return eval(s)
        else:
            return s
    file = os.path.join(config.dir.storage_root_dir, config.dir.storage_root.benchmark, 'rt_bench.xlsx')
    benchmark = pd.read_excel(file)
    benchmark['ground_truth'] = benchmark['ground_truth'].apply(lambda x: eval_string(x))
    benchmark['ground_truth_debatable'] = benchmark['ground_truth_debatable'].apply(lambda x: eval_string(x))
    benchmark['dist_val'] = benchmark['dist_val'].apply(lambda x: eval_string(x))
    benchmark = benchmark[['header', 'ground_truth', 'ground_truth_debatable', 'dist_val_count', 'dist_val']]
    CORPUS_NAME = 'labeled_benchmark'
    return benchmark

def load_excel_benchmark():
    global CORPUS_NAME
    def eval_string(s):
        if type(s) == str:
            return eval(s)
        else:
            return s
    file = os.path.join(config.dir.storage_root_dir, config.dir.storage_root.benchmark, 'st_bench.xlsx')
    benchmark = pd.read_excel(file)
    benchmark['ground_truth'] = benchmark['ground_truth'].apply(lambda x: eval_string(x))
    benchmark['ground_truth_debatable'] = benchmark['ground_truth_debatable'].apply(lambda x: eval_string(x))
    benchmark['dist_val'] = benchmark['dist_val'].apply(lambda x: eval_string(x))
    benchmark = benchmark[['header', 'ground_truth', 'ground_truth_debatable', 'dist_val_count', 'dist_val']]
    CORPUS_NAME = 'excel_benchmark'
    return benchmark

def load_labeled_benchmark_20percent():
    global CORPUS_NAME
    def eval_string(s):
        if type(s) == str:
            return eval(s)
        else:
            return s
    file = os.path.join(config.dir.storage_root_dir, config.dir.storage_root.benchmark, 'benchmark_20percent.xlsx')
    benchmark = pd.read_excel(file)
    benchmark['ground_truth'] = benchmark['ground_truth'].apply(lambda x: eval_string(x))
    benchmark['ground_truth_debatable'] = benchmark['ground_truth_debatable'].apply(lambda x: eval_string(x))
    benchmark['dist_val'] = benchmark['dist_val'].apply(lambda x: eval_string(x))
    benchmark = benchmark[['header', 'ground_truth', 'ground_truth_debatable', 'dist_val_count', 'dist_val']]
    CORPUS_NAME = 'labeled_benchmark_20percent'
    return benchmark

def load_labeled_benchmark_10percent():
    global CORPUS_NAME
    def eval_string(s):
        if type(s) == str:
            return eval(s)
        else:
            return s
    file = os.path.join(config.dir.storage_root_dir, config.dir.storage_root.benchmark, 'benchmark_10percent.xlsx')
    benchmark = pd.read_excel(file)
    benchmark['ground_truth'] = benchmark['ground_truth'].apply(lambda x: eval_string(x))
    benchmark['ground_truth_debatable'] = benchmark['ground_truth_debatable'].apply(lambda x: eval_string(x))
    benchmark['dist_val'] = benchmark['dist_val'].apply(lambda x: eval_string(x))
    benchmark = benchmark[['header', 'ground_truth', 'ground_truth_debatable', 'dist_val_count', 'dist_val']]
    CORPUS_NAME = 'labeled_benchmark_10percent'
    return benchmark

def load_labeled_benchmark_5percent():
    global CORPUS_NAME
    def eval_string(s):
        if type(s) == str:
            return eval(s)
        else:
            return s
    file = os.path.join(config.dir.storage_root_dir, config.dir.storage_root.benchmark, 'benchmark_5percent.xlsx')
    benchmark = pd.read_excel(file)
    benchmark['ground_truth'] = benchmark['ground_truth'].apply(lambda x: eval_string(x))
    benchmark['ground_truth_debatable'] = benchmark['ground_truth_debatable'].apply(lambda x: eval_string(x))
    benchmark['dist_val'] = benchmark['dist_val'].apply(lambda x: eval_string(x))
    benchmark = benchmark[['header', 'ground_truth', 'ground_truth_debatable', 'dist_val_count', 'dist_val']]
    CORPUS_NAME = 'labeled_benchmark_5percent'
    return benchmark

def load_excel_benchmark_20percent():
    global CORPUS_NAME
    def eval_string(s):
        if type(s) == str:
            return eval(s)
        else:
            return s
    file = os.path.join(config.dir.storage_root_dir, config.dir.storage_root.benchmark, 'Excel_benchmark_20percent.xlsx')
    benchmark = pd.read_excel(file)
    benchmark['ground_truth'] = benchmark['ground_truth'].apply(lambda x: eval_string(x))
    benchmark['ground_truth_debatable'] = benchmark['ground_truth_debatable'].apply(lambda x: eval_string(x))
    benchmark['dist_val'] = benchmark['dist_val'].apply(lambda x: eval_string(x))
    benchmark = benchmark[['header', 'ground_truth', 'ground_truth_debatable', 'dist_val_count', 'dist_val']]
    CORPUS_NAME = 'excel_benchmark_20percent'
    return benchmark

def load_excel_benchmark_10percent():
    global CORPUS_NAME
    def eval_string(s):
        if type(s) == str:
            return eval(s)
        else:
            return s
    file = os.path.join(config.dir.storage_root_dir, config.dir.storage_root.benchmark, 'Excel_benchmark_10percent.xlsx')
    benchmark = pd.read_excel(file)
    benchmark['ground_truth'] = benchmark['ground_truth'].apply(lambda x: eval_string(x))
    benchmark['ground_truth_debatable'] = benchmark['ground_truth_debatable'].apply(lambda x: eval_string(x))
    benchmark['dist_val'] = benchmark['dist_val'].apply(lambda x: eval_string(x))
    benchmark = benchmark[['header', 'ground_truth', 'ground_truth_debatable', 'dist_val_count', 'dist_val']]
    CORPUS_NAME = 'excel_benchmark_10percent'
    return benchmark

def load_excel_benchmark_5percent():
    global CORPUS_NAME
    def eval_string(s):
        if type(s) == str:
            return eval(s)
        else:
            return s
    file = os.path.join(config.dir.storage_root_dir, config.dir.storage_root.benchmark, 'Excel_benchmark_5percent.xlsx')
    benchmark = pd.read_excel(file)
    benchmark['ground_truth'] = benchmark['ground_truth'].apply(lambda x: eval_string(x))
    benchmark['ground_truth_debatable'] = benchmark['ground_truth_debatable'].apply(lambda x: eval_string(x))
    benchmark['dist_val'] = benchmark['dist_val'].apply(lambda x: eval_string(x))
    benchmark = benchmark[['header', 'ground_truth', 'ground_truth_debatable', 'dist_val_count', 'dist_val']]
    CORPUS_NAME = 'excel_benchmark_5percent'
    return benchmark


def load_len500():
    global CORPUS_NAME
    def eval_string(s):
        if type(s) == str:
            return eval(s)
        else:
            return s
    file = os.path.join(config.dir.storage_root_dir, config.dir.storage_root.synth_dataset, 'len500.txt')
    header = ['col_header', 'dist_val_count', 'dist_val']
    schema = {
        'col_header': str,
        'dist_val_count' : int,
        'dist_val' : str
    }
    df = pd.read_csv(file, dtype = schema, header = None, names = header, sep = '\t', usecols = ['col_header', 'dist_val_count', 'dist_val'], on_bad_lines ='skip')
    df['dist_val'] = df['dist_val'].apply(lambda x: eval_string(x))
    CORPUS_NAME = 'len500'
    return df.reset_index()

def load_data_cleaning_benchmark():
    global CORPUS_NAME
    csv_file = os.path.join(config.dir.storage_root_dir, 'data_cleaning_benchmark/data_cleaning_benchmark.txt')
    header = ['fname', 'col_header', 'dist_val_count', 'dist_val_str']
    schema = {
        'fname': str,
        'col_header': str,
        'dist_val_str' : str,
        'dist_val_count' : int
    }

    corpus = pd.read_csv(csv_file, dtype = schema, header = None, names = header, sep = '\t', usecols = ['fname', 'col_header', 'dist_val_str', 'dist_val_count'], on_bad_lines ='skip')
    corpus['col_header'].fillna('', inplace=True)
    corpus['dist_val'] = corpus['dist_val_str'].apply(lambda x: str(x).split("___"))
    corpus['dist_val'] = corpus['dist_val'].apply(lambda x: [] if not type(x) == list else x)
    del corpus['dist_val_str']
    CORPUS_NAME = 'data_cleaning_benchmark'
    return corpus

def load_data_cleaning_benchmark_groundtruth():
    csv_file = os.path.join(config.dir.storage_root_dir, 'data_cleaning_benchmark/data_cleaning_benchmark_groundtruth.txt')
    header = ['fname', 'col_header', 'dist_val_count', 'dist_val_str']
    schema = {
        'fname': str,
        'col_header': str,
        'dist_val_str' : str,
        'dist_val_count' : int
    }

    corpus = pd.read_csv(csv_file, dtype = schema, header = None, names = header, sep = '\t', usecols = ['fname', 'col_header', 'dist_val_str', 'dist_val_count'], on_bad_lines ='skip')
    corpus['col_header'].fillna('', inplace=True)
    corpus['dist_val'] = corpus['dist_val_str'].apply(lambda x: str(x).split("___"))
    corpus['dist_val'] = corpus['dist_val'].apply(lambda x: [] if not type(x) == list else x)
    del corpus['dist_val_str']
    return corpus
