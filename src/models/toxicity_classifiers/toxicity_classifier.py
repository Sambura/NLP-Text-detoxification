import torch
import typing
from transformers import BatchEncoding

class ToxicityClassifier():
    """
    Abstract base class for toxicity evaluation models
    """
    def __init__(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None

    def predict(self, 
                input: typing.Union[str, list[str], BatchEncoding, dict], 
                to_toxic: bool=False, 
                threshold: float=0.5) -> typing.Union[torch.Tensor, bool, list]:
        """
        Get a prediction for input(s)

        Parameters:
        input: Input to the model. Can be a string, a list of strings, or 
            BatchEncoding|dict with 'index_ids' and 'attention_mask' keys present
        to_toxic (bool): If True, binary classification is performed. If false, the
            method retuns raw model outputs
        threshold (float): If the model is a regressor, this threshold determines, 
            which model predictions to consider toxic 

        Returns:
        if input was a single string, returns model's output as list[float], or bool, if 
            to_toxic is True
        Otherwise, torch.Tensor with model's output is retunred, or torch.Tensor
            of booleans, if to_toxic is True
        """
        raise NotImplementedError()
    
    def __call__(self, input, **kwargs):
        """Same as predict()"""
        return self.predict(input, **kwargs)

    def collate_batch(self, batch): 
        """Collate function for batching"""
        return self.tokenizer.pad(batch, return_tensors='pt').to(self.device)

    def to(self, device: torch.device):
        """Sends the model to the specified device"""
        self.device = device
        self.model.to(device)
        return self