from torch.optim import Adam,SGD
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF

class Model(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.class_num = self.config["class_num"]
        self.bert = BertModel.from_pretrained(r"E:\BaiduNetdiskDownload\第六周 语言模型\bert-base-chinese\bert-base-chinese",return_dict=False)
        self.classify = nn.Linear(self.bert.config.hidden_size,self.class_num)
        self.crf = CRF(self.class_num,batch_first=True)

    def forward(self,x,target=None,attention_mask=None):
        x,_ = self.bert(x,attention_mask=attention_mask)
        x = self.classify(x)

        if target is not None:
            mask = target.gt(-1)
            return - self.crf(x,target,mask, reduction="mean")
        else:
            return self.crf.decode(x)
        


def choose_optim(config,model):
    lr = config["learning_rate"]
    if(config["optimizer"]=="adam"):
        return Adam(model.parameters(),lr=lr)
    elif(config["optimizer"]=="sgd"):
        return SGD(model.parameters(),lr=lr)
