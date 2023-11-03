import urllib.request
import zipfile
import os

def download_data(
        data_url:str="https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip", 
        data_dest:str="./data/",
        zip_name:str="compressed.zip") -> None:
    """
    Download dataset from the specified url and extract it in the given directory

    Parameters:
    data_url (str): Url to the zip archive to download
    data_dest (str): Path to directory where to put the downloaded file and where the files will be extracted.
        If the directory doesn't exist, it will be automatically created
    zip_name (str): How the downloaded archive should be named
    """
    zip_destination = os.path.join(data_dest, zip_name)

    os.makedirs(os.path.dirname(zip_destination), exist_ok=True)

    if not os.path.exists(zip_destination):
        urllib.request.urlretrieve(data_url, zip_destination)

    with zipfile.ZipFile(zip_destination, 'r') as zip_ref:
        zip_ref.extractall(data_dest)
