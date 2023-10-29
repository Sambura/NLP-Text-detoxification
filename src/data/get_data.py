import urllib.request
import zipfile
import typing
import os

def get_data(
        data_url:typing.Optional[str]=None, 
        data_dest:typing.Optional[str]=None, 
        zip_name:typing.Optional[str]=None):
    """
        Download dataset and extract it
    """
    data_url = data_url or "https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip"
    data_destination = data_dest or "./data/raw/"
    zip_name = zip_name or "compressed.zip"
    zip_destination = os.path.join(data_destination, zip_name)

    os.makedirs(os.path.dirname(zip_destination), exist_ok=True)

    if not os.path.exists(zip_destination):
        urllib.request.urlretrieve(data_url, zip_destination)

    with zipfile.ZipFile(zip_destination, 'r') as zip_ref:
        zip_ref.extractall(data_destination)
