from torch.utils.data import random_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import GenerationConfig, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

try:
    from ..data.make_dataset import load_detoxification_dataset
    from ..utils.training_utils import seed_everything
except ImportError:
    import sys
    if '.' not in sys.path: sys.path.append('.')
    from src.data.make_dataset import load_detoxification_dataset
    from src.utils.training_utils import seed_everything

class DetoxifierTrainer():
    def __init__(self, seed=None):
        seed_everything(seed)
        self.pretrained_name = 't5-small'
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.history_path = 'models/t5-detoxifier-history'

    def load_pretrained(self, pretrained_name='t5-small'):
        self.pretrained_name = pretrained_name
        self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

    def load_dataset(self, path='data/raw/filtered.tsv',
                    cache_path='data/interim/',
                    val_ratio=0.2,
                    dataset_portion=1,
                    verbose=False):
        if self.tokenizer is None: self.load_pretrained()
        self.train_dataset, self.val_dataset = \
            load_detoxification_dataset(
                path, 
                cache_path, 
                self.tokenizer, 
                val_split=val_ratio, 
                portion=dataset_portion, 
                verbose=verbose
            )

    def get_default_generation_config(self):
        genConfig = GenerationConfig.from_pretrained(self.pretrained_name)
        genConfig.max_new_tokens = 64
        return genConfig

    def get_default_training_args(self, generation_config=None, batch_size=32, epochs=10):
        if generation_config is None: generation_config = self.get_default_generation_config()
        
        return Seq2SeqTrainingArguments(
            self.history_path,
            evaluation_strategy = "epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.04,
            save_total_limit=5,
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

    def train(self, model_save_path, trainer=None, verbose=False):
        if verbose: print('Seed applied')
        if trainer is None: trainer = self.make_trainer(verbose=verbose)
        if verbose: print('Trainer created')
        self.model_save_path = model_save_path

        trainer.train()
        trainer.save_model(self.model_save_path)

def main(model_save_path='models/t5-detoxifier', dataset_portion=1, seed=1984, verbose=True):
    if verbose: print('Default training procedure...')
    trainer = DetoxifierTrainer(seed=seed)
    trainer.load_dataset(dataset_portion=dataset_portion, verbose=verbose)

    seed_everything(seed)
    trainer.train(model_save_path, verbose=verbose)
    if verbose: print(f'Training finished. The model is written to {trainer.model_save_path}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("train_model")
    parser.add_argument('-p', '--portion', default=1, type=float)
    args = parser.parse_args()
    main(dataset_portion=args.portion)
