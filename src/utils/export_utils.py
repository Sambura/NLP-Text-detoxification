from pathlib import Path
import pandas as pd
import os

def export_dataframe(df: pd.DataFrame, path: str, create_dirs: bool=True, sep: str='\t', index: bool=False) -> None:
    """
    Saves a dataframe to the specified file

    Parameters:
    df (pd.DataFrame): The dataframe to export
    path (str): Export path
    create_dirs (bool): If True, creates all necessary parent directories, if they don't exist already
    sep (str): Delimiter to use when saving the dataframe
    index (bool): If True, dataframe index gets exported along with the data 
    """
    if create_dirs:
        export_path_parent = Path(path).parent.absolute()
        os.makedirs(export_path_parent, exist_ok=True)
    df.to_csv(path, sep=sep, index=index)
