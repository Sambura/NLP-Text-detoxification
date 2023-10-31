import pandas as pd
from pathlib import Path
import os
import json
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

def download_if_needed(path):
    if not os.path.exists(path):
        parent = Path(path).parent.absolute()
        download_data(data_dest=parent)

def load_data(path, drop_columns=True, sort_toxicity=True, flatten=False):
    download_if_needed(path)
    df = pd.read_csv(path, delimiter='\t', index_col=0)
    if drop_columns: drop_extra_columns_inplace(df)
    if sort_toxicity: sort_by_toxicity_inplace(df)
    
    if flatten: return flatten_data(df)
    return df

def load_tokenized_data(path, cache_path, tokenizer, max_length=128, drop_columns=True, sort_toxicity=True, flatten=False):
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, delimiter='\t')
        df['reference'] = [json.loads(x) for x in df['reference']]
        df['translation'] = [json.loads(x) for x in df['translation']]
    else:
        download_if_needed(path)
        df = pd.read_csv(path, delimiter='\t', index_col=0)
        df = tokenize_data(df, tokenizer, max_length)
        cache_path_parent = Path(cache_path).parent.absolute()
        os.makedirs(cache_path_parent, exist_ok=True)
        df.to_csv(cache_path, sep='\t', index=False)

    if drop_columns: drop_extra_columns_inplace(df)
    if sort_toxicity: sort_by_toxicity_inplace(df)
    
    if flatten: return flatten_data(df)
    return df
