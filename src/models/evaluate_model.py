from tqdm.auto import tqdm
import torch
import pandas as pd
import numpy as np
import gc

from toxicity_classifiers.t5_toxicity_evaluator import T5TEModel
from toxicity_classifiers.roberta_toxicity_classifier import RTCModel

class DetoxifierEvaluator():
    def __init__(self, eval_model):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = eval_model
        self.model.to(self.device)

    def collate_fn(self, texts):
        return self.model.tokenizer(texts, return_tensors='pt', padding=True).to(self.device)

    def evaluate(self, texts, batch_size=128):
        dataloader = torch.utils.data.DataLoader(texts, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
        evals = []

        for batch in tqdm(dataloader, total=len(dataloader), desc='Evaluation'):
            output = self.model(batch, to_toxic=True)
            evals += output.detach().cpu()

        return evals

def get_random_rows(df, portion=1):
    return df[:][np.random.rand(len(df.index)) <= portion]

def main(predictions_path, use_roberta=False, weights_path='models/t5-toxicity-regressor/model.pt', portion=1, verbose=True):
    if verbose: print('Loading model...')
    model = RTCModel() if use_roberta else T5TEModel(weights_path)
    evaluator = DetoxifierEvaluator(model)

    if verbose: print('Loading data...')
    df = pd.read_csv(predictions_path, sep='\t')
    df = get_random_rows(df, portion)
    batch_size = 32 if use_roberta else 128
    ref_evals = evaluator.evaluate(df['Input'].astype(str).tolist(), batch_size)
    gc.collect()
    torch.cuda.empty_cache()
    trn_evals = evaluator.evaluate(df['Detoxified version'].astype(str).tolist(), batch_size)

    ttr = np.sum(ref_evals)
    ttt = np.sum(trn_evals)
    print(f'Evaluation results according to {"RoBERTa classifier" if use_roberta else "T5 regressor"}:')
    print(f'Total toxic references: {ttr}')
    print(f'Total toxic translations: {ttt}')
    print(f'{100 * (1 - ttt / ttr) : 0.2f}% samples detoxified succesfully')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("evaluate_model")
    parser.add_argument('-p', '--portion', default=1, type=float)
    parser.add_argument('--use_roberta', action='store_true')
    parser.add_argument('-d', '--predictions_path', default='data/predicted/predictions.tsv', type=str)
    args = parser.parse_args()
    main(args.predictions_path, use_roberta=args.use_roberta, portion=args.portion)
