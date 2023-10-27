from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.tokenization_utils_base import BatchEncoding
import torch
from torch import nn

class T5ToxicityRegressor(nn.Module):
    def __init__(self, encoder):
        super(T5ToxicityRegressor, self).__init__()
        self.encoder = encoder
        self.h1 = nn.Linear(512, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        x = self.h1(encoded)

        return nn.functional.sigmoid(torch.sum(x[:,:,0] * attention_mask, dim=1))

class T5TEModel():
    def __init__(self, weights_path=None):
        model_checkpoint = "t5-small"
        t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        encoder = t5_model.encoder
        
        self.model = T5ToxicityRegressor(encoder)
        if weights_path is not None:
            weights = torch.load(weights_path)
            self.model.load_state_dict(weights)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.device = torch.device('cpu')

    def predict(self, input):
        if isinstance(input, (BatchEncoding, dict)):
            return self.model(input_ids=input['input_ids'], attention_mask=input['attention_mask'])
        elif isinstance(input, list):
            tokenized = self.tokenizer.encode(input, padding=True, return_tensors='pt').to(self.device)
            return self.model(input_ids=tokenized['input_ids'], attention_mask=tokenized['attention_mask'])
        elif isinstance(input, str):
            tokenized = self.tokenizer(input, return_tensors='pt')
            return self.model(input_ids=tokenized['input_ids'], attention_mask=tokenized['attention_mask']).detach().numpy().tolist()[0]
        
        return None
    
    def __call__(self, input): return self.predict(input)

    def collate_batch(self, batch): return self.tokenizer.pad(batch, return_tensors='pt').to(self.device)

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self