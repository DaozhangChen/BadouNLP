import torch.nn as nn
import torch
from transformers import BertModel


class LanguageModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config["bert_path"],return_dict=False)
        self.drop_out = nn.Dropout(p=0.1)
        self.classify = nn.Linear(self.bert.config.hidden_size,self.bert.config.vocab_size)
        self.loss = nn.functional.cross_entropy
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self,x,y=None,mask=None,loss_mask_index=None):
        x = self.bert(x,attention_mask=mask)[0]
        x = self.drop_out(x)
        x = self.classify(x)
        if y is not None :
            new_x = torch.FloatTensor([]).to(x.device)
            new_y = torch.LongTensor([]).to(y.device)
            for input,target,loss_index in zip(x,y,loss_mask_index):
               new_x = torch.cat((new_x,input[loss_index[0]:,:]),dim=0) 
               new_y = torch.cat((new_y,target[loss_index[0]:]),dim=0)
            return self.loss(new_x.reshape(-1,x.shape[-1]),new_y.reshape(-1))
        else:
            return self.softmax(x)[0][-1]



def choose_optimizer(config,model,lr):
    if config["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(),lr=lr)
    elif config["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(),lr=lr)
