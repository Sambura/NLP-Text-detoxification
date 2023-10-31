from torch import nn
import torch

class T5ToxicityRegressor(nn.Module):
    def __init__(self, encoder):
        super(T5ToxicityRegressor, self).__init__()
        self.encoder = encoder
        self.h1 = nn.Linear(512, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        x = self.h1(encoded)

        return nn.functional.sigmoid(torch.sum(x[:,:,0] * attention_mask, dim=1))