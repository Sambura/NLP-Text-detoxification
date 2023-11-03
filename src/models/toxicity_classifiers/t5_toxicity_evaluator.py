from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.tokenization_utils_base import BatchEncoding
import torch
from .toxicity_classifier import ToxicityClassifier
from .t5_toxicity_regressor.regressor import T5ToxicityRegressor

class T5TEModel(ToxicityClassifier):
    """
    Interface for predicting using T5-based toxicity regressor
    """
    def __init__(self, weights_path: str=None):
        """
        Initialize this model from the specified weights
        """
        super().__init__()
        model_checkpoint = "t5-small"

        # We need an encoder to create the model, which could probably be made
        # smarter, but oh well
        t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        encoder = t5_model.encoder
        
        self.model = T5ToxicityRegressor(encoder).eval()
        if weights_path is not None: # just don't question this please
            weights = torch.load(weights_path)
            self.model.load_state_dict(weights)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def predict(self, input, to_toxic=False, threshold=0.5):
        if isinstance(input, (BatchEncoding, dict)):
            output = self.model(input_ids=input['input_ids'], attention_mask=input['attention_mask'])
        elif isinstance(input, list):
            tokenized = self.tokenizer.encode(input, padding=True, return_tensors='pt').to(self.device)
            output = self.model(input_ids=tokenized['input_ids'], attention_mask=tokenized['attention_mask'])
        elif isinstance(input, str): 
            tokenized = self.tokenizer(input, return_tensors='pt')
            output = self.model(input_ids=tokenized['input_ids'], attention_mask=tokenized['attention_mask']).detach().numpy().tolist()
            return output[0] > threshold if to_toxic else output[0]
        else:
            raise RuntimeError('Unsupported input type')
        
        return output > threshold if to_toxic else output
