from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import GenerationConfig, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import typing

try:
    from ..data.make_dataset import load_detoxification_dataset
    from ..utils.training_utils import seed_everything
except ImportError:
    import sys
    if '.' not in sys.path: sys.path.append('.')
    from src.data.make_dataset import load_detoxification_dataset
    from src.utils.training_utils import seed_everything

class DetoxifierTrainer():
    """
    This class contains methods for training the detoxification text-to-text model

    Attributes:
    model (PretrainedModelForSeq2SeqLM): the model itself
    tokenizer (PretrainedTokenizer): the tokenizer for the model
    train_dataset (list): Train dataset used for model training
    val_dataset (list): Validaton dataset used for model training
    """
    def __init__(self, seed: typing.Optional[int]=None) -> None:
        """
        Initialize this DetoxifierTrainer and optionally seed random generators
        """
        seed_everything(seed)
        self.pretrained_name = 't5-small'
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.history_path = 'models/t5-detoxifier-history'

    def load_pretrained(self, pretrained_name: str='t5-small') -> None:
        """
        Load pretrained model and tokenizer by its name
        """
        self.pretrained_name = pretrained_name
        self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

    def load_dataset(self, path: str='data/raw/filtered.tsv',
                    cache_path: str='data/interim/',
                    val_ratio: float=0.2,
                    dataset_portion: float=1,
                    verbose: bool=False) -> None:
        """
        Loads the dataset from the specified file(s). If there is no file
        at the specified path, it gets downloaded. The datasets are stored
        in self.train_dataset and self.val_dataset attributes

        Parameters:
        path (str): The path where the dataset file is/should be located
        cache_path (str): The path to the directory where the tokenized data
            should be cached, or path to the pretokenized .tsv file
        val_ratio (float): the fraction of the dataset to put in the validation set
        dataset_portion (float): the fraction of dataset to use
        verbose (bool): if True, prints progress messages
        """
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

    def get_default_generation_config(self) -> GenerationConfig:
        """
        Get the default generation config for the model
        """
        genConfig = GenerationConfig.from_pretrained(self.pretrained_name)
        genConfig.max_new_tokens = 64
        return genConfig

    def get_default_training_args(self, 
                                  generation_config: typing.Optional[GenerationConfig]=None, 
                                  batch_size: int=32, 
                                  epochs: int=10) -> Seq2SeqTrainingArguments:
        """
        Get default training argument for the model trainer

        Parameters:
        generation_config (GenerationConfig): Generation config to use during training
        batch_size (int): batch_size for training
        epochs (int): number of training epochs

        Returns:
        The Seq2SeqTrainingArguments object with resulting training arguments
        """
        if generation_config is None: generation_config = self.get_default_generation_config()
        
        return Seq2SeqTrainingArguments(
            self.history_path,
            evaluation_strategy = "epoch",
            learning_rate=1e-3,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=5,
            num_train_epochs=epochs,
            predict_with_generate=True,
            fp16=True,
            report_to='tensorboard',
            logging_steps=5000,
            save_steps=10000,
            generation_config=generation_config
        )

    def make_trainer(self, 
                     args: typing.Optional[Seq2SeqTrainingArguments]=None, 
                     collator: typing.Optional[typing.Callable]=None, 
                     verbose: bool=False) -> Seq2SeqTrainer:
        """
        Construct a Seq2SeqTrainer object to train the model

        Parameters:
        args (Seq2SeqTrainingArguments): custom training arguments for the trainer
        collator (function): custom data collator
        verbose (bool): if True, prints progress messages

        Returns:
        Seq2SeqTrainer object which can be used for model training
        """
        if self.model is None: self.load_pretrained() # load the model
        if verbose: print('Pretrained model loaded')
        if self.train_dataset is None: self.load_dataset() # load the data
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

    def train(self, 
              model_save_path: str, 
              trainer: typing.Optional[Seq2SeqTrainer]=None, 
              verbose: bool=False) -> None:
        """
        Train the model

        Parameters:
        model_save_path (str): Location where the final model should be saved
        trainer (Seq2SeqTrainer): Custom trainer
        verboes (bool): If True, prints progress messages
        """
        if verbose: print('Seed applied')
        if trainer is None: trainer = self.make_trainer(verbose=verbose)
        if verbose: print('Trainer created')
        self.model_save_path = model_save_path

        trainer.train()
        trainer.save_model(self.model_save_path)

def main(model_save_path: str='models/t5-detoxifier', dataset_portion: float=1, seed: int=1984, verbose: bool=True) -> None:
    """
    Starts the standard model training procedure

    Parameters:
    model_save_path (str): The path where the final model should be saved
    dataset_portion (float): The fraction of dataset to use for training (to speed up training use less data)
    seed (int): Seed for random generation
    verbose (bool): If True, prints progress messages
    """
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
