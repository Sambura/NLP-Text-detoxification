from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers.tokenization_utils_base import BatchEncoding
import torch
from .toxicity_classifier import ToxicityClassifier

class RTCModel(ToxicityClassifier):
    def __init__(self):
        self.model = RobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
        self.tokenizer = RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')

    def predict(self, input, to_toxic=False, threshold=0.5):
        if isinstance(input, (BatchEncoding, dict)):
            output = self.model(input_ids=input['input_ids'], attention_mask=input['attention_mask']).logits
        elif isinstance(input, list):
            tokenized = self.tokenizer.encode(input, padding=True, return_tensors='pt').to(self.device)
            output = self.model(input_ids=tokenized['input_ids'], attention_mask=tokenized['attention_mask']).logits
        elif isinstance(input, str):
            output = self.model(self.tokenizer.encode(input, return_tensors='pt')).logits.detach().numpy().tolist()[0]
            return output[1] > output[0] if to_toxic else output
        else:
            raise RuntimeError('Unsupported input type')
        
        return output[:,1] if to_toxic else output
