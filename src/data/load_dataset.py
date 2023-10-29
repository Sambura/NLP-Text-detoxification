from .load_data import load_tokenized_data, flatten_data
from random import shuffle
import numpy as np

def make_detoxification_dataset(df):
    return [{'input_ids': ref, 'labels': trn } for ref, trn in zip(df['reference'], df['translation'])]

def make_toxicity_dataset(df):
    if 'text' not in df.columns: df = flatten_data(df)
    return [{'input_ids': ids, 'labels': tox} for ids, tox in zip(df['text'], df['toxicity'])]

def get_random_portion(dataset, portion):
    shuffle(dataset)
    return dataset[:int(np.clip(portion, 0, 1) * len(dataset))]

def load_detoxification_dataset(path, cache_path, tokenizer, portion=1):
    """
        Detoxification dataset contains pairs of (reference, translation)
        to train a model to translate toxic text into neutral one. The format of the data is:
        { 'input_ids': ref_input_ids, 'labels': tranldation_input_ids }
    """
    df = load_tokenized_data(path, cache_path, tokenizer)
    dataset = make_detoxification_dataset(df)
    
    if portion == 1: return dataset
    return get_random_portion(dataset, portion)

def load_toxicity_dataset(path, cache_path, tokenizer, portion=1):
    """
        Toxicity dataset contains pairs of (text, toxicity) to train a model
        to evaluate toxicity of a given text. The format of the data is:
        { 'input_ids': input_ids, 'labels': toxicity }
    """
    df = load_tokenized_data(path, cache_path, tokenizer)
    dataset = make_toxicity_dataset(df)

    if portion == 1: return dataset
    return get_random_portion(dataset, portion)

# dataset_kwargs = {
#     'path': 'data/raw/filtered.tsv', # path to raw data
#     'cache_path': 'data/processed/tokenized.tsv', # path to processed data (or where to store it)
#     'tokenizer': None, # tokenizer to tokenize texts
#     'portion': 0.001 # get only a portion of dataset [0..1]
# }
# print(load_detoxification_dataset(**dataset_kwargs))