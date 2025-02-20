import json
from config import Config
import random
from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader


class DataGenerator:
    def __init__(self,config,train_type="normal"):
        self.config = config
        self.train_type = train_type
        self.data_path = self.config["data_path"]
        self.origin_data = []
        self.corpus = ""
        self.tokenizer = BertTokenizer.from_pretrained(self.config["bert_path"])
        self.load()

    def load(self):
        if self.train_type == "sft":
            with open(self.data_path,encoding="utf8") as f:
                for line in f:
                    data = json.loads(line)
                    self.origin_data.append(data)
        elif self.train_type == "normal":
            with open(self.config["corpus_path"],encoding="gbk") as f:
                for line in f:
                    self.corpus += line.strip()
                self.corpus_length = len(self.corpus)

    
    def get_sample(self):
        if self.train_type == "sft":
            random_sentence = random.choice(self.origin_data)
            title = random_sentence["title"]
            content = random_sentence["content"]
            max_length = self.config["max_length"]
            input_title = title
            target_title = title[1:]
            input = input_title + "[SEP]" + content
            target = target_title + "[SEP]" + content
            
            input = self.tokenizer.encode(input,max_length=max_length,padding="max_length",truncation=True,add_special_tokens=False)
            target = self.tokenizer.encode(target,max_length=max_length,padding="max_length",truncation=True,add_special_tokens=False)
            sep_token_index = input.index(102) ##49 48 -1
            loss_mask_index = sep_token_index  - max_length + 1
            mask = self.set_attention_mask(sep_token_index)

            return [torch.LongTensor(input),torch.LongTensor(target),torch.FloatTensor(mask),torch.LongTensor([loss_mask_index])]
        elif self.train_type == "normal":
            start_index = random.randint(0,self.corpus_length-self.config["max_length"]-2)
            input = self.corpus[start_index:start_index+self.config["max_length"]]
            target = self.corpus[start_index + 1:start_index+self.config["max_length"]+1]
            input = self.tokenizer.encode(input,max_length=self.config["max_length"],padding="max_length",truncation=True)
            target = self.tokenizer.encode(target,max_length=self.config["max_length"],padding="max_length",truncation=True)
            mask = torch.tril(torch.ones(self.config["max_length"],self.config["max_length"]))
            mask = mask.masked_fill(mask==0,float("-inf"))
            
            return [torch.LongTensor(input),torch.LongTensor(target),torch.FloatTensor(mask)]

    
    def set_attention_mask(self,sep_index):
        # 2
        max_length = self.config["max_length"]
        a = torch.ones((sep_index,sep_index))
        b = torch.zeros((sep_index,max_length-sep_index))
        c = torch.ones((max_length-sep_index,sep_index))
        d = torch.tril(torch.ones(max_length-sep_index,max_length-sep_index))
        ab = torch.cat((a,b),dim=1)
        cd = torch.cat((c,d),dim=1)
        mask = torch.cat((ab,cd),dim=0)
        mask = mask.masked_fill(mask==0,float("-inf"))
        return mask

    def __len__(self):
        return self.config["epoch_data_number"]
    
    def __getitem__(self,idx):
        return self.get_sample()
    


def get_data(config,train_type="normal"):
    dg = DataGenerator(config,train_type=train_type)
    dl = DataLoader(dg,batch_size=config["batch_size"],shuffle=True)
    return dl

        




if __name__ == "__main__":
    dl = get_data(Config,train_type="sft")

    for data in dl:
        print(data)
        break;
        