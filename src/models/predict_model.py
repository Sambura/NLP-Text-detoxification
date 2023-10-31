from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
from tqdm.auto import tqdm
import torch
import pandas as pd
import numpy as np
import os
from pathlib import Path

try:
    from ..data.make_dataset import load_toxicity_dataset
except ImportError:
    import sys
    sys.path.append('.')
    from src.data.make_dataset import load_toxicity_dataset


class DetoxifierPredictor():
    def __init__(self, path='models/trained_detoxifier', model=None, tokenizer=None):
        self.model_path = path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path) if tokenizer is None else tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path) if model is None else model
        self.model.eval()
        self.model.config.use_cache = False
        self.model.to(self.device)

    def get_generation_config(self, max_new_tokens=128):
        genConfig = GenerationConfig.from_pretrained(self.model_path)
        genConfig.max_new_tokens = max_new_tokens
        return genConfig
        
    def translate_text(self, inference_request):
        tokenized = self.tokenizer.encode(inference_request, return_tensors="pt").to(self.device)
        gen_config = self.get_generation_config(256)
        outputs = self.model.generate(tokenized, generation_config=gen_config)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def collate_func(self, batch):
        return self.tokenizer.pad(batch, return_tensors='pt').to(self.device)

    def get_eval_dataset(self, path='data/raw/filtered.tsv', cache_path='data/interim/tokenized.tsv', dataset_portion=1):
        return load_toxicity_dataset(path, cache_path, self.tokenizer, portion=dataset_portion)

    def get_dataloader(self, dataset, batch_size=128):
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_func)

    def predict(self, dataloader=None):
        transformed = []
        gen_config = self.get_generation_config()

        for batch in tqdm(dataloader, total=len(dataloader), desc='Translating'):
            output = self.model.generate(input_ids=batch.input_ids, attention_mask=batch.attention_mask, generation_config=gen_config)
            transformed += output.detach().cpu()
        
        transformed = [{'input_ids': x} for x in transformed]
        return transformed
    
    def decode(self, transformed):
        texts = []

        for sample in tqdm(transformed, desc='Decoding'):
            texts.append(self.tokenizer.decode(sample['input_ids'], skip_special_tokens=True))

        return texts
    
    def export_ref_trn_dataframe(self, decoded_refs, decoded_trns, export_path='data/predicted/predictions.tsv'):
        df = pd.DataFrame(np.array([decoded_refs, decoded_trns]).T, columns=['Input', 'Detoxified version'])
        export_path_parent = Path(export_path).parent.absolute()
        os.makedirs(export_path_parent, exist_ok=True)
        df.to_csv(export_path, sep='\t', index=False)
    
def main(model_path='models/t5_detoxifier-10x10lr', dataset_portion=1, verbose=True):
    if verbose: print('Loading model...')
    predictor = DetoxifierPredictor(model_path)
    if verbose: print('Loading data...')
    dataset = predictor.get_eval_dataset(dataset_portion=dataset_portion)
    dataloader = predictor.get_dataloader(dataset)
    translations = predictor.predict(dataloader)
    decoded_translations = predictor.decode(translations)
    decoded_inputs = predictor.decode(dataset)
    if verbose: print('Exporting...')
    predictor.export_ref_trn_dataframe(decoded_inputs, decoded_translations)

if __name__ == '__main__':
    main()
    