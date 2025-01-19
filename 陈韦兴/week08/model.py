import torch.nn as nn
import torch
from config import Config
class Sentence_Embedding(nn.Module):
    def __init__(self,Config):
        super().__init__()
        self.config = Config
        self.vocab_size = self.config["vocab_size"]
        self.hidden_size = self.config["hidden_size"]
        self.embedding = nn.Embedding(self.vocab_size,self.hidden_size,padding_idx=0)
        self.linear = nn.Linear(self.hidden_size,self.hidden_size)
        self.max_length = self.config["max_length"]
        if self.config["pool_type"] == "avg":
            self.pool = nn.AvgPool1d(self.max_length)
        else:
            self.pool = nn.MaxPool1d(self.max_length)

    def forward(self,x):
        x = self.embedding(x)
        x = self.linear(x)
        x = torch.transpose(x,1,2)
        x = self.pool(x)
        x = torch.squeeze(x)
        return x
    

class Calc_Vector_Loss(nn.Module):
    def __init__(self, Config):
        super().__init__()
        self.config = Config
        self.sentence_embedding = Sentence_Embedding(self.config)
        self.loss = nn.TripletMarginLoss()
    
    def forward(self,anchor,positive=None,negative=None):
        if positive is not None and negative is not None:
            anchor_vector = self.sentence_embedding(anchor)
            positive_vector = self.sentence_embedding(positive)
            negative_vector = self.sentence_embedding(negative)
            return self.loss(anchor_vector,positive_vector,negative_vector)
        else:
            return self.sentence_embedding(anchor)


def choose_optimizer(config,model):
    if config["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(),lr=config["learning_rate"])
    elif(config["optimizer"] == "sgd"):
        return torch.optim.SGD(model.parameters(),lr=config["learning_rate"])
        

    

if __name__ == "__main__":
    Config["vocab_size"] = 11111
    # a = Sentence_Embedding(Config)
    b = Calc_Vector_Loss(Config)
    test = torch.tensor([[1,2,3,4,5,6,7,8,9,0]])
    test2 = torch.tensor([[11,21,31,41,51,61,71,81,91,10]])
    test3 = torch.tensor([[12,22,32,42,52,62,72,82,92,20]])
    print(b(test,test2,test3))
