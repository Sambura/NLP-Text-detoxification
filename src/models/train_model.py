import transformers
import torch
import random
import numpy as np
from torch.utils.data import random_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import GenerationConfig, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

try:
    from ..data.make_dataset import load_detoxification_dataset
except ImportError:
    import sys
    sys.path.append('.')
    from src.data.make_dataset import load_detoxification_dataset

def seed_everything(seed):
    if seed is None: return None
    transformers.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None

class DetoxifierTrainer():
    def __init__(self):
        self.pretrained_name = 't5-small'
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.history_path = 'models/detoxifier_history'

    def load_pretrained(self, pretrained_name='t5-small'):
        self.pretrained_name = pretrained_name
        self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

    def load_dataset(self, path='data/raw/filtered.tsv',
                    cache_path='data/interim/tokenized.tsv',
                    val_ratio=0.2,
                    dataset_portion=1):
        if self.tokenizer is None: self.load_pretrained()
        dataset = load_detoxification_dataset(path, cache_path, self.tokenizer, dataset_portion)
        self.train_dataset, self.val_dataset = random_split(dataset, [1 - val_ratio, val_ratio])

    def get_default_generation_config(self):
        genConfig = GenerationConfig.from_pretrained(self.pretrained_name)
        genConfig.max_new_tokens = 64
        return genConfig

    def get_default_training_args(self, generation_config=None, batch_size=32, epochs=5):
        if generation_config is None: generation_config = self.get_default_generation_config()
        
        return Seq2SeqTrainingArguments(
            self.history_path,
            evaluation_strategy = "epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=epochs,
            predict_with_generate=True,
            fp16=True,
            report_to='tensorboard',
            logging_steps=5000,
            save_steps=10000,
            generation_config=generation_config
        )

    def make_trainer(self, args=None, collator=None, verbose=False):
        if self.model is None: self.load_pretrained()
        if verbose: print('Pretrained model loaded')
        if self.train_dataset is None: self.load_dataset()
        if verbose: print('Dataset loaded')
        if args is None: args = self.get_default_training_args()
        if collator is None: collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        return Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=collator,
            tokenizer=self.tokenizer,
        )

    def train(self, model_save_path, trainer=None, seed=None, verbose=False):
        seed_everything(seed)
        if verbose: print('Seed applied')
        if trainer is None: trainer = self.make_trainer(verbose=verbose)
        if verbose: print('Trainer created')
        self.model_save_path = model_save_path

        trainer.train()
        trainer.save_model(self.model_save_path)

def main(model_save_path='models/trained_detoxifier', dataset_portion=1, seed=1984, verbose=True):
    if verbose: print('Default training procedure...')
    trainer = DetoxifierTrainer()
    trainer.load_dataset(dataset_portion=dataset_portion)
    trainer.train(model_save_path, seed=seed, verbose=verbose)
    if verbose: print(f'Training finished. The model is written to {trainer.model_save_path}')

if __name__ == '__main__':
    main(dataset_portion=0.1)
