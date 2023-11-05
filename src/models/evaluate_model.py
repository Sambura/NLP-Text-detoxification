from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import gc
import os

from typing import Optional

try:
    from .toxicity_classifiers.toxicity_classifier import ToxicityClassifier
    from .toxicity_classifiers.t5_toxicity_evaluator import T5TEModel
    from .toxicity_classifiers.roberta_toxicity_classifier import RTCModel
except ImportError:
    import sys
    if '.' not in sys.path: sys.path.append('.')
    from src.models.toxicity_classifiers.toxicity_classifier import ToxicityClassifier
    from src.models.toxicity_classifiers.t5_toxicity_evaluator import T5TEModel
    from src.models.toxicity_classifiers.roberta_toxicity_classifier import RTCModel


class DetoxifierEvaluator():
    "Class for evaluating detoxifier model predictions"
    def __init__(self, eval_model: ToxicityClassifier) -> None:
        """Initialize this evaluator with the given toxicity evaluator/classifier"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = eval_model
        self.model.to(self.device)

    def collate_fn(self, texts):
        return self.model.tokenizer(texts, return_tensors='pt', padding=True).to(self.device)

    def evaluate(self, texts: list[str], batch_size: int=128) -> list[bool]:
        """
        Evaluates the toxicity of the given texts.

        Parameters:
        texts: (list[str]): list of texts to classify: toxic vs. neutral
        batch_size (int): batch size used during evaluation

        Returns:
        list of bools, if the i'th element is True, then i'th element of `texts` is toxic,
        according to the model
        """
        dataloader = torch.utils.data.DataLoader(texts, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
        evals = []

        for batch in tqdm(dataloader, total=len(dataloader), desc='Evaluation'):
            output = self.model(batch, to_toxic=True)
            evals += output.detach().cpu()

        return evals

def get_random_rows(df: pd.DataFrame, portion: float=1) -> pd.DataFrame:
    # pretty sure there's a function for that...
    return df[:][np.random.rand(len(df.index)) <= portion]

def main(predictions_path: str, 
         use_roberta: bool=False, 
         weights_path: Optional[str]=None, 
         portion: float=1, 
         verbose: bool=True,
         export_path: Optional[str]=None,
         batch_size: Optional[int]=None) -> None:
    """
    Runs default evaluation procedure

    Parameters:
    predictions_path (str): The path to the file containing model's predictions, 
        exported by predict_model.py
    use_roberta (bool): if True, use RoBERTa-based toxicity classifier. Otherwise,
        use T5-based toxicity regressor for evaluation
    weights_path (str): Path to the model's weights (only required when `use_roberta` == False)
    portion (float): The portion of the predictions to evaluate
    verbose (bool): If True, messages reporting progress will be printed
    """
    if verbose: print('Loading model...')
    model = RTCModel() if use_roberta else T5TEModel(weights_path)
    evaluator = DetoxifierEvaluator(model)

    if verbose: print('Loading data...')
    df = pd.read_csv(predictions_path, sep='\t')
    df = get_random_rows(df, portion)
    # roberta seems to be MUCH heavier, so can't afford large batch size
    if batch_size is None: batch_size = 32 if use_roberta else 128
    ref_evals = evaluator.evaluate(df['reference'].astype(str).tolist(), batch_size)
    # this can be removed, but then probably use batch_size = 4 for roberta or something
    gc.collect()
    torch.cuda.empty_cache()
    trn_evals = evaluator.evaluate(df['translation'].astype(str).tolist(), batch_size)

    ttr = np.sum(ref_evals) # number of toxic samples in inputs
    ttt = np.sum(trn_evals) # number of toxic samples in outputs
    print(f'Evaluation results according to {"RoBERTa classifier" if use_roberta else "T5 regressor"}:')
    print(f'Total toxic references: {ttr}')
    print(f'Total toxic translations: {ttt}')
    print(f'{100 * (1 - ttt / ttr) :0.2f}% samples detoxified succesfully')

    if export_path is not None:
        df = pd.DataFrame(
            np.array([df['reference'], df['translation'], np.array(ref_evals, dtype=bool), np.array(trn_evals, dtype=bool)]).T, 
            columns=['reference', 'translation', 'ref_tox', 'trn_tox']
        )
        export_path_parent = Path(export_path).parent.absolute()
        os.makedirs(export_path_parent, exist_ok=True)
        df.to_csv(export_path, sep='\t', index=False)
        print(f'Exported evalutaions to `{export_path}`')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("evaluate_model")
    parser.add_argument('-p', '--portion', default=1, type=float)
    parser.add_argument('--use_roberta', action='store_true')
    parser.add_argument('-w', '--weights_path', default='models/t5-toxicity-regressor/model.pt')
    parser.add_argument('-d', '--predictions_path', default='data/predicted/predictions.tsv', type=str)
    args = parser.parse_args()
    main(args.predictions_path, use_roberta=args.use_roberta, weights_path=args.weights_path, portion=args.portion)
