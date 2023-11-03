from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase
from tqdm.auto import tqdm
import torch
import pandas as pd
import numpy as np
import os
from pathlib import Path
import typing

try: # there is probably a proper way to do it but presumably this info is kept secret
    from ..data.make_dataset import load_toxicity_dataset
except ImportError:
    import sys
    sys.path.append('.')
    from src.data.make_dataset import load_toxicity_dataset

class DetoxifierPredictor():
    """
    Class for using the T5 detoxifier to detoxify text!

    Attributes:
    model (PreTrainedModel): The actual model
    tokenizer: The tokenizer
    device: The pytorch device
    """
    def __init__(self, path: str, model: PreTrainedModel=None, tokenizer: PreTrainedTokenizerBase=None) -> None:
        """
        Initialize this predictor with the pretrained model

        Parameters:
        path (str): Path to the model to load
        model (PreTrainedModel, optional): The model, in case you already have it loaded
        tokenizer (PreTrainedTokenizer, optional): The tokenizer, in case you already have it loaded
        """
        self.model_path = path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path) if tokenizer is None else tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path) if model is None else model
        self.model.eval()
        self.model.config.use_cache = False
        self.model.to(self.device)

    def get_generation_config(self, max_new_tokens: int=128) -> GenerationConfig:
        """
        Get default text generation configuration for this model

        Parameters:
        max_new_tokens (int): Modify the max number of generated tokens
        """
        # I didn't figure out how to clone this config, so I have to use model_path
        genConfig = GenerationConfig.from_pretrained(self.model_path)
        genConfig.max_new_tokens = max_new_tokens
        return genConfig
        
    def translate_text(self, inference_request: str) -> str:
        """Transform the given text by the detoxification model"""
        tokenized = self.tokenizer.encode(inference_request, return_tensors="pt").to(self.device)
        gen_config = self.get_generation_config(256)
        outputs = self.model.generate(tokenized, generation_config=gen_config)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def collate_func(self, batch):
        return self.tokenizer.pad(batch, return_tensors='pt').to(self.device)

    def get_eval_dataset(self, path: str, cache_path: str, dataset_portion: float=1) -> list:
        """
        Loads the dataset for running prediction on.

        Parameters:
        Refer to load_toxicity_dataset() function
        """
        return load_toxicity_dataset(path, cache_path, self.tokenizer, portion=dataset_portion)

    def get_dataloader(self, dataset: list, batch_size: int=128) -> torch.utils.data.DataLoader:
        """Create a dataloader for the given dataset"""
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_func)

    def predict(self, dataloader: torch.utils.data.DataLoader) -> list[dict[str, list[int]]]:
        """
        Run prediction on the data in the given dataloader

        Returns:
        The list of predictions. 
        Each prediction is a dict with a single element: { 'input_ids': predicted_tokens }
        """
        transformed = []
        gen_config = self.get_generation_config()

        for batch in tqdm(dataloader, total=len(dataloader), desc='Translating'):
            output = self.model.generate(input_ids=batch.input_ids, attention_mask=batch.attention_mask, generation_config=gen_config)
            transformed += output.detach().cpu()
        
        transformed = [{'input_ids': x} for x in transformed]
        return transformed
    
    def decode(self, transformed: list[dict[str, list[int]]]) -> list[str]:
        """
        Decode the list of tokenized texts, i.e. ones returned by DetoxifierPredictor.predict()

        Parameters:
        transformed (list): list of data to decode. Each element should contain a key 'input_ids'
            with value being a list of token ids

        Returns:
        list of decoded texts
        """
        texts = []

        for sample in tqdm(transformed, desc='Decoding'):
            texts.append(self.tokenizer.decode(sample['input_ids'], skip_special_tokens=True))

        return texts
    
    def export_ref_trn_dataframe(self, decoded_refs: list[str], decoded_trns: list[str], export_path: str) -> None:
        """
        Export predictions to a .tsv file along with the input data.

        Parameters:
        decoded_refs (list[str]): list of inputs to the model in a text form
        decoded_trns (list[str]): list of outputs from the model in a text form
        export_path (str): path where the file should be exported
        """
        df = pd.DataFrame(np.array([decoded_refs, decoded_trns]).T, columns=['Input', 'Detoxified version'])
        export_path_parent = Path(export_path).parent.absolute()
        os.makedirs(export_path_parent, exist_ok=True)
        df.to_csv(export_path, sep='\t', index=False)
    
    
def main(model_path: str, 
         dataset_path: str, 
         tokenized_path: str, 
         export_path: str, 
         dataset_portion: float=1, 
         translate_str: typing.Optional[str]=None, 
         verbose: bool=True) -> None:
    """
    This function does a prediction on a given dataset using the T5 detoxifier model

    Parameters:
    model_path (str): Path to the saved T5 detoxifier model
    dataset_path (str): The path to the raw dataset
    tokenized_path (str): The path to the tokenized dataset. If the file exists, this
        parameter overrides `dataset_path`
    export_path (str): Filename for model's predictions (.tsv format)
    dataset_portion (float): The fraction of the dataset to run predictions on
    translate_str (str): If not None, makes this function transform this string using the
        model, print it, and return from function. In this case only all parameters except
        `model_path` are optional (can be set to None)
    verbose (bool): If True, progress report messages are printed
    """
    if verbose: print('Loading model...')
    predictor = DetoxifierPredictor(model_path)

    if translate_str is not None:
        print(f'Translation for `{translate_str}`:')
        print(predictor.translate_text(translate_str))
        return

    if verbose: print('Loading data...')
    dataset = predictor.get_eval_dataset(dataset_path, tokenized_path, dataset_portion=dataset_portion)
    dataloader = predictor.get_dataloader(dataset)
    translations = predictor.predict(dataloader)
    decoded_translations = predictor.decode(translations)
    decoded_inputs = predictor.decode(dataset)
    if verbose: print('Exporting...')
    predictor.export_ref_trn_dataframe(decoded_inputs, decoded_translations, export_path)
    if verbose: print(f'Exported predictions to: {export_path}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("predict_model")
    parser.add_argument('-m', '--model_path', default='models/t5-detoxifier', type=str)
    parser.add_argument('-t', '--translate', dest='prompt', default=None, type=str)
    parser.add_argument('-d', '--dataset_path', default=None, type=str)
    parser.add_argument('--tokenized_path', default=None, type=str)
    parser.add_argument('-o', '--export_path', default='data/predicted/predictions.tsv', type=str)
    parser.add_argument('-p', '--portion', help='Portion of dataset to run prediction on (rows will be randomly selected)', default=1, type=float)

    args = parser.parse_args()
    if args.dataset_path is None: 
        args.dataset_path = 'data/raw/filtered.tsv'
        if args.tokenized_path is None: args.tokenized_path = 'data/interim/tokenized.tsv'

    main(
        args.model_path, 
        args.dataset_path, 
        args.tokenized_path, 
        args.export_path,
        args.portion,
        args.prompt
    )
