from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, PreTrainedTokenizerBase
from torch.utils.data import random_split
from torch import nn
import torch
import os
import typing

try: # relative imports refuse to work if the script is run as __main__
    from ....data.make_dataset import load_toxicity_dataset
    from ....utils.training_utils import seed_everything
    from .regressor import T5ToxicityRegressor
except ImportError:
    import sys
    if '.' not in sys.path: sys.path.append('.')
    from src.data.make_dataset import load_toxicity_dataset
    from src.utils.training_utils import seed_everything
    from src.models.toxicity_classifiers.t5_toxicity_regressor.regressor import T5ToxicityRegressor

class RegressorTrainer(Trainer):
    """
    Trainer with MSE loss. You have to set self.MSE attribute with
    the instance of nn.MSELoss
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = self.MSE(outputs, inputs.pop("labels"))
        return (loss, outputs) if return_outputs else loss

def get_collate_fn(tokenizer: PreTrainedTokenizerBase) -> typing.Callable:
    """
    Get collate_fn function for model training
    """
    def collate_batch(batch):
        return tokenizer.pad(batch, return_tensors='pt')
    return collate_batch

def main(output_path: str, portion: float=1, verbose: bool=True) -> None:
    """
    Perform T5-based toxicity regressor model training

    Parameters:
    output_path (str): The path where the model should be stored
    portion (float): The fraction of the dataset to use for training
    verbose (bool): If True, prints progress messages
    """
    seed_everything(seed=1984)

    model_checkpoint = "t5-small"
    if verbose: print('Loading model...')
    # The toxicity regressor is based on T5 model encoder, so we get that:
    encoder = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).encoder
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    if verbose: print('Loading data...')
    # this loads both toxic and neutral text samples along with their toxicity scores
    dataset = load_toxicity_dataset(
        path='data/raw/filtered.tsv',
        cache_path='data/interim/tokenized.tsv',
        tokenizer=tokenizer,
        include_translations=True,
        portion=portion
    )

    val_ratio = 0.2
    train_dataset, val_dataset = random_split(dataset, [1 - val_ratio, val_ratio])

    model = T5ToxicityRegressor(encoder)

    batch_size = 64
    training_args = TrainingArguments(
        output_path,
        evaluation_strategy = "epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.03,
        save_total_limit=10,
        num_train_epochs=10,
        logging_steps=3000,
        save_steps=5000,
        fp16=True,
        report_to='tensorboard',
    )

    # I use custom trainer so that I can use MSELoss
    trainer = RegressorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=get_collate_fn(tokenizer),
    )
    # I didn't want to mess up Trainer's constructor, so I just set this attribute here
    trainer.MSE = nn.MSELoss() 

    trainer.train()

    torch.save(model.state_dict(), os.path.join(output_path, 'model.pt'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("evaluate_model")
    parser.add_argument('-p', '--portion', default=1, type=float)
    parser.add_argument('-o', '--output_path', default='models/t5-toxicity-regressor/', type=str)
    parser.add_argument('-q', '--quiet', action='store_true')
    args = parser.parse_args()
    main(args.output_path, portion=args.portion, verbose=not args.quiet)
