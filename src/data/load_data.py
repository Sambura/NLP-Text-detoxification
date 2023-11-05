import pandas as pd
from pathlib import Path
import os
import json
import numpy as np
from transformers import PreTrainedTokenizerBase
import typing
from urllib.error import URLError
from .download_data import download_data

def drop_extra_columns_inplace(df: pd.DataFrame) -> None:
    """
    Drops columns `similarity` and `length_diff` from the given dataframe inplace.
    """
    df.drop(columns=['similarity', 'lenght_diff'], inplace=True, errors='ignore')

def sort_by_toxicity_inplace(df: pd.DataFrame) -> None:
    """
    Sorts the contents of columns `reference`, `translation`, and `ref_tox`, `trn_tox` of the given 
    dataframe inplace. After this, value form `ref_tox` is guaranteed to be greater than value from `trn_tox`
    from the same row. The text in `reference` and `translation` columns is also sorted accordingly
    """
    mask = df['ref_tox'] < df['trn_tox']
    df.loc[mask, ['translation', 'reference']] = df.loc[mask, ['reference', 'translation']].values
    df.loc[mask, ['ref_tox', 'trn_tox']] = df.loc[mask, ['trn_tox', 'ref_tox']].values

def flatten_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flattens the detoxification dataset: reference and translation coluns are combined into
    a `text` column, and ref_tox & trn_tox combined into `toxicity`.

    Returns:
    A flattened copy of the given dataframe
    """
    merge_rename_dict = {'translation': 'reference', 'trn_tox': 'ref_tox'}
    rename_dict = {'reference': 'text', 'ref_tox': 'toxicity'}
    renamed = df[['translation', 'trn_tox']].rename(columns=merge_rename_dict)
    flat = pd.concat((df[['reference', 'ref_tox']], renamed), axis=0)
    flat.reset_index(drop=True, inplace=True)
    return flat.rename(columns=rename_dict)

def tokenize_data(df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, max_length: int=256) -> pd.DataFrame:
    """
    Tokenize the text in the given dataframe. Tokenizes `reference` and `translation` columns. The resulting
    dataframe contains lists of integers (token ids) instead of text

    Parameters:
    df (DataFrame): the dataframe with text to tokenize
    tokenizer (PreTrainedTokenizer): the tokenizer to use
    max_length (int): the max length for the tokenized text

    Returns:
    A copy of the initial dataframe with tokenized text
    """
    tokenized_df = df.copy()

    def tokenize_column(col):
        column = tokenized_df[col].tolist()
        tokenized_df[col] = tokenizer(column, max_length=max_length, truncation=True).input_ids
        
    tokenize_column('reference')
    tokenize_column('translation')

    return tokenized_df

def download_if_needed(path: str, verbose: bool=False, num_tries: int=3) -> None:
    """
    Downloads the raw dataset file if it does not already exist.

    Parameters:
    path (str): the path where the downloaded data should be
    verbose (bool): if True, prints progress messages
    num_tries (int): number of times to try to download the data in case of failure
    """
    if not os.path.exists(path):
        for i in range(num_tries):
            try:
                if verbose: print('Downloading raw data...')
                parent = Path(path).parent.absolute()
                download_data(data_dest=parent)
                break
            except URLError:
                if verbose: print(f'Download failed, retry {i + 1}')

def preprocess_dataframe(df: pd.DataFrame, 
                         drop_columns: bool=True, 
                         sort_toxicity: bool=True, 
                         flatten: bool=False) -> pd.DataFrame:
    """
    Apply preprocessing to the detoxification dataset dataframe. This function may
    modify the original dataframe.

    Parameters:
    df (pd.DataFrame): The dataframe to preprocess
    drop_columns (bool): If True, drop `similarity` and `length_diff` columns
    sort_toxicity (bool): If True, sort text and toxicity columns as described in 
        `sort_by_toxicity_inplace` function
    flatten (bool): If True, flattens the dataframe as described in `flatten_data`
    
    Returns:
    The preprocessed DataFrame 
    """
    if drop_columns: drop_extra_columns_inplace(df)
    if sort_toxicity: sort_by_toxicity_inplace(df)
    if flatten: df = flatten_data(df)

    return df

def load_data(path: str, 
              drop_columns: bool=True, 
              sort_toxicity: bool=True, 
              flatten: bool=False) -> pd.DataFrame:
    """
    Load the detoxification dataset and apply specified preprocessing steps.
    If the dataset is not present, it gets downloaded.

    Parameters:
    path (str): The path where the dataset file is/should be located
    drop_columns (bool): passed to preprocess_dataframe()
    sort_toxicity (bool): passed to preprocess_dataframe()
    flatten (bool): passed to preprocess_dataframe()

    Returns:
    The preprocessed detoxificatoin DataFrame 
    """
    download_if_needed(path)
    df = pd.read_csv(path, delimiter='\t')

    return preprocess_dataframe(df, drop_columns, sort_toxicity, flatten)

def df_str_to_list_inplace(df: pd.DataFrame, columns: list[str]=['reference', 'translation']) -> None:
    """
    Convert string representations of lists to python lists in specified dataframe columns.

    Parameters:
    df (pd.DataFrame): A dataframe to process
    columns (list[str]): the columns to process
    """
    for col_name in columns:
        df[col_name] = [json.loads(x) for x in df[col_name]]

def split_data(df: pd.DataFrame, 
               cache_path: typing.Optional[str], 
               val_split: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataframe into a training and validation dataframes, storing
    the resulting split to files.

    Parameters:
    df (pd.DataFrame): The dataframe to split
    cache_path (str): The path to the directory where the resulting dataframes should be saved.
        If the directory already contains the split files, they are loaded instead. 
    val_split (float): The fraction of the dataset to put in validation dataframe

    Returns:
    Tuple of two dataframes: (train_dataframe, val_dataframe)
    """
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

def load_tokenized_data(path: str, 
                        cache_path: str, 
                        tokenizer: PreTrainedTokenizerBase, 
                        max_length: int=128, 
                        drop_columns: bool=True, 
                        sort_toxicity: bool=True, 
                        flatten: bool=False, 
                        verbose: bool=False, 
                        val_split: typing.Optional[float]=None)-> \
        typing.Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Load the detoxification dataset, tokenize the text, and apply specified 
    preprocessing steps. If the dataset is not present, it gets downloaded.
    The tokenized data is cached in file.

    Parameters:
    path (str): The path where the dataset file is/should be located
    cache_path (str): The path to the directory where the tokenized dataframe
        should be cached, or path to the pretokenized .tsv file
    tokenizer (PreTrainedTokenizer): the tokenizer for text tokenization
    max_length (int): the max length of tokenized text
    drop_columns (bool): passed to preprocess_dataframe()
    sort_toxicity (bool): passed to preprocess_dataframe()
    flatten (bool): passed to preprocess_dataframe()
    verbose (bool): if True, prints progress messages
    val_split (float): if not None, the fraction of data to put in validation dataset

    Returns:
    The preprocessed detoxificatoin DataFrame with tokenized text, or tuple of two 
    dataframes: (train_dataframe, val_dataframe)
    """
    tokenized_path = cache_path
    if tokenized_path is not None:
        if os.path.isfile(tokenized_path):
            cache_path = Path(tokenized_path).parent.absolute()
        else:
            tokenized_path = os.path.join(cache_path, 'tokenized.tsv')

    if tokenized_path is not None and os.path.exists(tokenized_path):
        if verbose: print('Loading tokenized data...')
        df = pd.read_csv(tokenized_path, delimiter='\t')
        df_str_to_list_inplace(df)
    else:
        download_if_needed(path, verbose)
        df = pd.read_csv(path, delimiter='\t')
        if verbose: print('Tokenizing...')
        df = tokenize_data(df, tokenizer, max_length)
        if tokenized_path is not None:
            if verbose: print('Backing up...')
            os.makedirs(cache_path, exist_ok=True)
            df.to_csv(tokenized_path, sep='\t', index=False)

    if val_split is not None:
        tdf, vdf = split_data(df, cache_path, val_split)

        return ( # copy needed to avoid pandas' warnings
            preprocess_dataframe(tdf.copy(), drop_columns, sort_toxicity, flatten), 
            preprocess_dataframe(vdf.copy(), drop_columns, sort_toxicity, flatten)
        )

    return preprocess_dataframe(df, drop_columns, sort_toxicity, flatten)
    