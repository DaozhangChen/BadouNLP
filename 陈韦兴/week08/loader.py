from collections import defaultdict
from config import Config
import json
import random
import torch
from torch.utils.data import DataLoader



class DataGenerator:
    def __init__(self,Config,data_type):
        self.config = Config
        self.data_type = data_type
        self.schema_dict = load_schema(Config["schema_path"])
        self.vocab_dict = load_vocab(Config["vocab_path"])
        self.train_data_dict = load_train_data(Config["train_path"])
        self.valid_data = load_valid_data(Config["valid_path"])
        self.config["vocab_size"] = len(self.vocab_dict)
        self.max_length = self.config["max_length"]
        self.encoder_sentence_dict = self.encoder_sentence()
        self.positive_ratio = self.config["positive_ratio"]
        
        
    def __len__(self):
        if self.data_type == "train":
            return self.config["epoch_data_size"]
        elif self.data_type == "valid":
            return len(self.valid_data)
        

    def __getitem__(self,idx):
        if self.data_type == "train":
            return self.set_train_data()
        elif self.data_type == "valid":
            return self.valid_data[idx]
        

    def encoder_sentence(self):
        encoder_sentence_dict = defaultdict(list)
        if self.data_type == "train":
            for index,title in enumerate(self.train_data_dict):
                encoder_title = self.schema_dict.get(title,None)
                questions = self.train_data_dict[title]
                for question in questions:
                    question = [self.vocab_dict.get(char,self.vocab_dict["[UNK]"]) for i,char in enumerate(question)]
                    question = padding(question,self.config["max_length"])
                    encoder_sentence_dict[encoder_title].append(question)
        elif self.data_type == "valid":
            for question_desc,question_type_name in self.valid_data:
                question_type = self.schema_dict[question_type_name]
                question = [self.vocab_dict.get(char,self.vocab_dict["[UNK]"]) for i,char in enumerate(question_desc)]
                question = padding(question,self.config["max_length"])
                encoder_sentence_dict[question_type].append(question)
        return encoder_sentence_dict
    
    def set_train_data(self):
        random_key = random.sample(list(self.encoder_sentence_dict.keys()),2)
        if len(list(self.encoder_sentence_dict[random_key[0]]))>1 and len(list(self.encoder_sentence_dict[random_key[1]]))>=1:
            positive_sample = random.sample(self.encoder_sentence_dict[random_key[0]],2)
            negative_sample = random.choice(self.encoder_sentence_dict[random_key[1]])

            return [torch.LongTensor(positive_sample[0]),torch.LongTensor(positive_sample[1]),torch.LongTensor(negative_sample)]
        else:
            return self.set_train_data()
        
    def get_vector_question_dict(self):
        return self.encoder_sentence_dict
    
    def get_schema_dict(self):
        return self.schema_dict
    
    def format_sentence(self,sentence):
        sentence_code = [self.vocab_dict.get(char,self.vocab_dict["[UNK]"]) for i,char in enumerate(sentence)]
        sentence_code = padding(sentence_code,self.config["max_length"])
        return sentence_code
        


def padding(char_list,max_length):
    new_char_list = char_list[:max_length]
    new_char_list += [0] * (max_length - len(new_char_list))
    return new_char_list




def load_vocab(path):
    vocab_dict = {}
    with open(path,encoding="utf8") as f:
        for index,line in enumerate(f) :
            token = line.strip()
            vocab_dict[token] = index + 1
    return vocab_dict
            


# 加载所有问题与问题对应的下标
def load_schema(path):
    with open(path,encoding="utf8") as f:
        data = json.load(f)
    return data

#加载训练数据
def load_train_data(path):
    train_data_dict = defaultdict(list)
    with open(path,encoding="utf8") as f:
        for line in f:
            line_data = json.loads(line)
            target = line_data["target"]
            questions = line_data["questions"]
            train_data_dict[target] = questions
    return train_data_dict

def load_valid_data(path):
    valid_data = []
    with open(path,encoding="utf8") as f:
        for line in f:
            line_data = json.loads(line) 
            valid_data.append(line_data)
    return valid_data
            



def load_data(config,data_type):
    dg = DataGenerator(config,data_type)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=True)
    return dl
            



if __name__ == "__main__":

    dl = load_data(Config,"train")

    
