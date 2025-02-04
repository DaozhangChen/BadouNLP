import json
import torch
from config import Config
from transformers import BertTokenizer
from torch.utils.data import DataLoader

class DataGenerator:
    def __init__(self,config,data_type):
        self.config = config
        self.max_length = self.config["max_length"]
        self.data_type = data_type
        self.tokenizer = BertTokenizer.from_pretrained(r"E:\BaiduNetdiskDownload\第六周 语言模型\bert-base-chinese\bert-base-chinese")
        self.schema = self.load_schema()
        self.data = []
        self.sentence = []
        self.load()
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        return self.data[idx]

    def load(self):
        if self.data_type == "train":
            self.load_data(self.config["train_data_path"])
        elif self.data_type == "valid":
            self.load_data(self.config["valid_data_path"])

    def load_schema(self):
        with open(self.config["schema_path"],encoding="utf8") as f:
            return json.load(f)
        
    def load_data(self,path):
        with open(path,encoding="utf8") as f:
            all_text = f.read()
            all_sentences = all_text.split("\n\n")

            for sentence in all_sentences:
                char_list = sentence.split("\n")
                current_char_list = []

                join_sentence = ""
                encode_input = []
                label_list = [8]
                
                for char_and_label in char_list:
                    char,label = char_and_label.split(" ")
                    char = char.strip()
                    label = label.strip()
                    current_char_list.append(char)
                    label_list.append(self.schema[label])
                    join_sentence += char
                label_list = self.padding(label_list)
                self.sentence.append("".join(current_char_list))

                token_dict = self.tokenizer(join_sentence,max_length=self.max_length,padding="max_length",truncation=True)
                encode_input = token_dict["input_ids"]
                attention_mask = token_dict["attention_mask"]
                self.data.append([torch.LongTensor(encode_input),torch.LongTensor(label_list),torch.LongTensor(attention_mask)])

    def padding(self,label_list):
        max_length = self.max_length
        label_list = label_list[:max_length]
        label_list += [-1] * (max_length - len(label_list))
        return label_list


def get_data_loader(config,data_type,shuffle=True):
    dg = DataGenerator(config,data_type)
    dl = DataLoader(dg,shuffle=shuffle,batch_size=config["batch_size"])
    return dl






if __name__ == "__main__":
    # loader = DataGenerator(Config,"valid")
    loader = get_data_loader(Config,"train")
    count = 0
    for data in loader:
        count += 1
        print(len(data[0]))
        
    print(count)