from .load_data import load_tokenized_data, flatten_data
from random import shuffle
import numpy as np

def make_detoxification_dataset(df):
    return [{'input_ids': ref, 'labels': trn } for ref, trn in zip(df['reference'], df['translation'])]

def make_toxicity_dataset(df, include_translations=False):
    if include_translations:
        if 'text' not in df.columns: df = flatten_data(df)
        return [{'input_ids': ids, 'labels': tox} for ids, tox in zip(df['text'], df['toxicity'])]
    
    return [{'input_ids': ids, 'labels': tox} for ids, tox in zip(df['reference'], df['ref_tox'])]

def get_random_portion(dataset, portion):
    if portion == 1: return dataset
    shuffle(dataset) # admittedly not the best way to do it but it's not like we're doing rocket science
    return dataset[:int(np.clip(portion, 0, 1) * len(dataset))]

def load_detoxification_dataset(path, cache_path, tokenizer, val_split=None, portion=1, verbose=False):
    """
        Detoxification dataset contains pairs of (reference, translation)
        to train a model to translate toxic text into neutral one. The format of the data is:
        { 'input_ids': ref_input_ids, 'labels': tranldation_input_ids }
    """
    dfs = load_tokenized_data(path, cache_path, tokenizer, verbose=verbose, val_split=val_split)
    if verbose: print("Making a dataset...")

    if val_split is not None:
        df_train, df_val = dfs
        dataset_train = make_detoxification_dataset(df_train)
        dataset_val = make_detoxification_dataset(df_val)
    
        return get_random_portion(dataset_train, portion), get_random_portion(dataset_val, portion)
    
    return get_random_portion(make_detoxification_dataset(dfs), portion)

def load_toxicity_dataset(path, cache_path, tokenizer, include_translations=False, portion=1):
    """
        Toxicity dataset contains pairs of (text, toxicity) to train a model
        to evaluate toxicity of a given text. The format of the data is:
        { 'input_ids': input_ids, 'labels': toxicity }
    """
    df = load_tokenized_data(path, cache_path, tokenizer)
    dataset = make_toxicity_dataset(df, include_translations)

    if portion == 1: return dataset
    return get_random_portion(dataset, portion)
