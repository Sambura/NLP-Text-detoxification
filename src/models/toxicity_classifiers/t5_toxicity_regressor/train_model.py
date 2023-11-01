from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from torch.utils.data import random_split
from torch import nn
import torch
import os

try:
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
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = self.MSE(outputs, inputs.pop("labels"))
        return (loss, outputs) if return_outputs else loss

def get_collate_fn(tokenizer):
    def collate_batch(batch):
        return tokenizer.pad(batch, return_tensors='pt')
    return collate_batch

def main(output_path='models/t5-toxicity-regressor/', portion=1, verbose=True):
    seed_everything(seed=1984)

    model_checkpoint = "t5-small"
    if verbose: print('Loading model...')
    encoder = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).encoder
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    if verbose: print('Loading data...')
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

    trainer = RegressorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=get_collate_fn(tokenizer),
    )
    trainer.MSE = nn.MSELoss()

    trainer.train()

    torch.save(model.state_dict(), os.path.join(output_path, 'model.pt'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("evaluate_model")
    parser.add_argument('-p', '--portion', default=1, type=float)
    args = parser.parse_args()
    main(portion=args.portion)
