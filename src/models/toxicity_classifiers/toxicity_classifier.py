import torch

class ToxicityClassifier():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None

    def predict(self, input, to_toxic=False, threshold=0.5):
        raise NotImplementedError()
    
    def __call__(self, input, **kwargs): 
        return self.predict(input, **kwargs)

    def collate_batch(self, batch): 
        return self.tokenizer.pad(batch, return_tensors='pt').to(self.device)

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self