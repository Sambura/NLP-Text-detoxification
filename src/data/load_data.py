import pandas as pd
from pathlib import Path
import os
import json
import numpy as np
from .download_data import download_data

def drop_extra_columns_inplace(df):
    df.drop(columns=["similarity", "lenght_diff"], inplace=True)

def sort_by_toxicity_inplace(df):
    mask = df['ref_tox'] < df['trn_tox']
    df.loc[mask, ['translation', 'reference']] = df.loc[mask, ['reference', 'translation']].values
    df.loc[mask, ['ref_tox', 'trn_tox']] = df.loc[mask, ['trn_tox', 'ref_tox']].values

def flatten_data(df):
    """
        Flattens the detoxification dataset: reference and translation coluns are combined into
        a `text` column, and ref_tox & trn_tox combined into `toxicity`. Flattens a copy of dataframe.
    """
    renamed = df[['translation', 'trn_tox']].rename(columns={'translation': 'reference', 'trn_tox': 'ref_tox'})
    flat = pd.concat((df[['reference', 'ref_tox']], renamed), axis=0).rename(columns={'reference': 'text', 'ref_tox': 'toxicity'})
    flat.reset_index(drop=True, inplace=True)
    return flat

def tokenize_data(df, tokenizer, max_length=128):
    tokenized_df = df.copy()

    def tokenize_column(col):
        column = tokenized_df[col].tolist()
        tokenized_df[col] = tokenizer(column, max_length=max_length, truncation=True).input_ids
        
    tokenize_column('reference')
    tokenize_column('translation')

    return tokenized_df

def download_if_needed(path, verbose=False):
    if not os.path.exists(path):
        if verbose: print('Downloading raw data...')
        parent = Path(path).parent.absolute()
        download_data(data_dest=parent)

def load_data(path, drop_columns=True, sort_toxicity=True, flatten=False):
    download_if_needed(path)
    df = pd.read_csv(path, delimiter='\t', index_col=0)
    if drop_columns: drop_extra_columns_inplace(df)
    if sort_toxicity: sort_by_toxicity_inplace(df)
    
    if flatten: return flatten_data(df)
    return df

def df_str_to_list_inplace(df, columns=['reference', 'translation']):
    for col_name in columns:
        df[col_name] = [json.loads(x) for x in df[col_name]]

def split_data(df, cache_path, val_split):
    total_len = len(df.index)

    val_len = int(total_len * val_split)
    train_len = total_len - val_len
    data_loaded = False
    
    if cache_path is not None:
        train_path = os.path.join(cache_path, f'train-{train_len}.tsv')
        val_path = os.path.join(cache_path, f'validation-{val_len}.tsv')

        if os.path.exists(train_path) and os.path.exists(val_path):
            train_df = pd.read_csv(train_path, delimiter='\t')
            val_df = pd.read_csv(val_path, delimiter='\t')
            df_str_to_list_inplace(train_df)
            df_str_to_list_inplace(val_df)
            data_loaded = True

    if not data_loaded:
        indices = np.arange(total_len)
        np.random.shuffle(indices)
        val_df = df.iloc[indices[:val_len]]
        train_df = df.iloc[indices[val_len:]]

    if cache_path is not None:
        os.makedirs(cache_path, exist_ok=True)
        train_df.to_csv(train_path, sep='\t', index=False)
        val_df.to_csv(val_path, sep='\t', index=False)

    return train_df, val_df

def load_tokenized_data(path, cache_path, tokenizer, max_length=128, drop_columns=True, sort_toxicity=True, flatten=False, verbose=False, val_split=None):
    tokenized_path = None if cache_path is None else os.path.join(cache_path, 'tokenized.tsv')
    if tokenized_path is not None and os.path.exists(tokenized_path):
        if verbose: print('Loading tokenized data...')
        df = pd.read_csv(tokenized_path, delimiter='\t')
        df_str_to_list_inplace(df)
    else:
        download_if_needed(path, verbose)
        df = pd.read_csv(path, delimiter='\t', index_col=0)
        if verbose: print('Tokenizing...')
        df = tokenize_data(df, tokenizer, max_length)
        if tokenized_path is not None:
            if verbose: print('Backing up...')
            os.makedirs(cache_path, exist_ok=True)
            df.to_csv(tokenized_path, sep='\t', index=False)

    if drop_columns: drop_extra_columns_inplace(df)
    if sort_toxicity: sort_by_toxicity_inplace(df)
    if flatten: df = flatten_data(df)

    if val_split is None:
        return df
    
    return split_data(df, cache_path, val_split)
