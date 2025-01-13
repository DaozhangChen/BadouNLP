import csv

from torch.utils.data import DataLoader
from transformers import BertTokenizer

import torch


class DataGenerator:
    def __init__(self, data_path, config, data_type):
        self.config = config
        # 加载词表
        self.vocab = self.load_vocab(config["vocab_path"])
        # 记录词表大小
        self.config["vocab_size"] = len(self.vocab)
        # 加载bert预训练模型
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        # 加载数据
        self.load(data_path, data_type)


    def load(self, path, data_type):

        train_data = []
        valid_data = []
        with open(path, encoding="utf8") as f:
            reader = csv.DictReader(f)
            count = 0
            for line in reader:
                # 将字用数字表示
                if self.config["model_type"] == "bert":
                    # 使用分词器
                    input_id = self.tokenizer.encode(line["review"].strip(), max_length=self.config["max_text_length"],
                                                     padding="max_length")
                else:
                    input_id = torch.LongTensor(self.encode_sentence(line["review"].strip()))
                # 将输出转为张量
                input_id = torch.LongTensor(input_id)
                input_label = torch.LongTensor([int(line["label"])])
                if count < 5:
                    train_data.append([input_id, input_label])
                    count += 1
                else:
                    valid_data.append([input_id, input_label])
                    count = 0
        if data_type == "train":
            self.data = train_data
        else:
            self.data = valid_data



    # 重写这两个方法，为了成为一个torch
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_vocab(self, path):
        vocab = {}
        with open(path, encoding="utf8") as f:
            for index, text in enumerate(f):
                text = text.strip()
                vocab[text] = index + 1
        return vocab

    def encode_sentence(self, sentence):
        input_id = []
        for word in sentence:
            input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_sentence_code):
        input_id = input_sentence_code[:self.config["max_text_length"]]
        input_id += [0] * (self.config["max_text_length"] - len(input_id))
        return input_id


def load_data(data_path, config, shuffle=True, data_type="train"):
    dg = DataGenerator(data_path, config,  data_type)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    dg = DataGenerator("文本分类练习.csv", Config,data_type="train")
    print(dg[1])
