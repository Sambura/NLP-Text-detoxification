from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers.tokenization_utils_base import BatchEncoding
import torch

class RTCModel():
    def __init__(self):
        self.model = RobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
        self.tokenizer = RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
        self.device = torch.device('cpu')

    def predict(self, input):
        if isinstance(input, (BatchEncoding, dict)):
            return self.model(input_ids=input['input_ids'], attention_mask=input['attention_mask']).logits
        elif isinstance(input, list):
            tokenized = self.tokenizer.encode(input, padding=True, return_tensors='pt').to(self.device)
            return self.model(input_ids=tokenized['input_ids'], attention_mask=tokenized['attention_mask']).logits
        elif isinstance(input, str):
            return self.model(self.tokenizer.encode(input, return_tensors='pt')).logits.detach().numpy().tolist()[0]
        
        return None
    
    def __call__(self, input): return self.predict(input)

    def collate_batch(self, batch): return self.tokenizer.pad(batch, return_tensors='pt').to(self.device)

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self