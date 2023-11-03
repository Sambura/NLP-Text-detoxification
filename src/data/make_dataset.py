from .load_data import load_tokenized_data, flatten_data
from random import sample
import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizerBase
import typing

def make_detoxification_dataset(df: pd.DataFrame) -> list[dict[str, list[int]]]:
    """
    Make a dataset for detoxification task from the given DataFrame.

    Parameters:
    df (dp.DataFrame): the dataframe containing `reference` and `translation` columns with
        tokenized text in them.
    
    Returns:
    List of dicts: {'input_ids': input_tokens, 'labels': label_tokens }
    """
    return [{'input_ids': ref, 'labels': trn } for ref, trn in zip(df['reference'], df['translation'])]

def make_toxicity_dataset(df: pd.DataFrame, include_translations: bool=False) -> list[dict]:
    """
    Make a dataset for toxicity regression task from the given DataFrame.

    Parameters:
    df (dp.DataFrame): the dataframe containing either `text` column or `reference` and `translation` 
        columns with tokenized text in them.
    include_translations (bool): if True, and the dataframe contains `translation` column, it gets
        included in the dataset, otherwise it is ignored
        
    Returns:
    List of dicts: {'input_ids': input_tokens, 'labels': toxicity }
    """
    if include_translations:
        if 'text' not in df.columns: df = flatten_data(df)
        return [{'input_ids': ids, 'labels': tox} for ids, tox in zip(df['text'], df['toxicity'])]
    
    return [{'input_ids': ids, 'labels': tox} for ids, tox in zip(df['reference'], df['ref_tox'])]

def get_random_portion(dataset: list, portion: float) -> list:
    """
    Randomly choose a sample of items from the dataset
    """
    if portion == 1: return dataset
    return sample(dataset, int(len(dataset) * portion))

def load_detoxification_dataset(path: str, 
                                cache_path: str, 
                                tokenizer: PreTrainedTokenizerBase, 
                                val_split: typing.Optional[float]=None, 
                                portion: float=1, 
                                verbose: bool=False) -> \
        typing.Union[list, tuple[list, list]]:
    """
    Loads the data from the disk and makes a detoxification dataset.

    Parameters:
    path (str): The path where the dataset file is/should be located
    cache_path (str): The path to the directory where the tokenized dataframe
        should be cached, or path to the pretokenized .tsv file
    tokenizer (PreTrainedTokenizer): the tokenizer for text tokenization
    val_split (float): if not None, the fraction of data to put in validation dataset
    portion (float): the fraction of data to use
    verbose (bool): if True, prints progress messages

    Returns:
    The detoxificatoin dataset, or tuple of two 
    dataset: (train_dataframe, val_dataframe)
    """
    dfs = load_tokenized_data(path, cache_path, tokenizer, verbose=verbose, val_split=val_split)
    if verbose: print("Making a dataset...")

    if val_split is not None:
        df_train, df_val = dfs
        dataset_train = make_detoxification_dataset(df_train)
        dataset_val = make_detoxification_dataset(df_val)
    
        return get_random_portion(dataset_train, portion), get_random_portion(dataset_val, portion)
    
    return get_random_portion(make_detoxification_dataset(dfs), portion)

def load_toxicity_dataset(path: str, 
                          cache_path: str, 
                          tokenizer: PreTrainedTokenizerBase, 
                          include_translations: bool=False, 
                          portion: float=1):
    """
    Loads the data from the disk and makes a toxicity evaluation dataset.

    Parameters:
    path (str): The path where the dataset file is/should be located
    cache_path (str): The path to the directory where the tokenized dataframe
        should be cached, or path to the pretokenized .tsv file
    tokenizer (PreTrainedTokenizer): the tokenizer for text tokenization
    val_split (float): if not None, the fraction of data to put in validation dataset
    portion (float): the fraction of data to use
    include_translations (bool): if True, includes contents of `translation` column in the dataset

    Returns:
    The toxicity evaluation dataset
    """
    df = load_tokenized_data(path, cache_path, tokenizer)
    dataset = make_toxicity_dataset(df, include_translations)

    if portion == 1: return dataset
    return get_random_portion(dataset, portion)
