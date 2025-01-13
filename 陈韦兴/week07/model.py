import torch
import torch.nn as nn
from transformers import BertModel,BertConfig
from torch.optim import SGD, Adam


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config['hidden_size']
        vocab_size = config['vocab_size'] + 1
        class_num = config['class_num']
        model_type = config['model_type']
        number_layers = config['number_layers']
        bert_config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=128,
            num_hidden_layers=1,
            num_attention_heads=8,
            intermediate_size=512,
            max_position_embeddings=512,
            return_dict=False
        )
        self.use_bert = False
        # 设置embedding层的初始化状态，设置0为padding的数值
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if model_type == 'fast_text':
            self.encoder = lambda x: x
        elif model_type == 'lstm':
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=number_layers, batch_first=True)
        elif model_type == "gru":
            self.encode = nn.GRU(hidden_size, hidden_size, number_layers=number_layers, batch_first=True)
        elif model_type == "rnn":
            self.encoder = nn.RNN(hidden_size, hidden_size, number_layers=number_layers, batch_first=True)
        elif model_type == "cnn":
            self.encoder = CNN(config)
        elif model_type == "gated_cnn":
            self.encoder = GatedCNN(config)
        elif model_type == "stack_gated_cnn":
            self.encoder = StackGatedCNN(config)
        elif model_type == "rcnn":
            self.encoder = RCNN(config)
        elif model_type == "bert":
            self.use_bert = True
            # self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
            self.encoder = BertModel(bert_config)
            hidden_size = self.encoder.config.hidden_size
        elif model_type == "bert_lstm":
            self.use_bert = True
            self.encoder = BertLSTM(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == "bert_cnn":
            self.use_bert = True
            self.encoder = BertCNN(config,bert_config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == "bert_mid_layer":
            self.use_bert = True
            self.encoder = BertMidLayer(config,bert_config)
            hidden_size = self.encoder.bert.config.hidden_size

        self.classify = nn.Linear(hidden_size, class_num)
        self.pooling_style = config['pooling_style']
        self.loss = nn.functional.cross_entropy

    def forward(self, x, target=None):
        if self.use_bert:
            x = self.encoder(x)
        else:
            x = self.embedding(x)
            x = self.encoder(x)

        if isinstance(x, tuple):
            x = x[0]

        if self.pooling_style == 'max':
            self.pooling_layer = nn.MaxPool1d(x.shape[1])
        else:
            self.pooling_layer = nn.AvgPool1d(x.shape[1])

        x = self.pooling_layer(x.transpose(1, 2)).squeeze()

        predict = self.classify(x)
        if target is not None:
            return self.loss(predict, target.squeeze())
        else:
            return predict


class CNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config["hidden_size"]
        kernel_size = config["kernel_size"]
        pad = int((kernel_size - 1) / 2)
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size, bias=False, padding=pad)

    def forward(self, x):
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)


class GatedCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cnn = CNN(config)
        self.gate = CNN(config)

    def forward(self, x):
        a = self.cnn(x)
        b = self.gate(x)
        b = torch.sigmoid(b)
        return torch.mul(a, b)


class StackGatedCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.number_layers = config["number_layers"]
        self.hidden_size = config["hidden_size"]
        self.gcnn_layers = nn.ModuleList(
            GatedCNN(config) for i in range(self.number_layers)
        )
        self.ff_liner_layers1 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.number_layers)
        )
        self.ff_liner_layers2 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.number_layers)
        )
        self.bn_after_gcnn = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.number_layers)
        )
        self.bn_after_ff = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.num_layers)
        )

    def forward(self, x):
        for i in range(self.number_layers):
            gcnn_x = self.gcnn_layers[i](x)
            x = gcnn_x + x
            x = self.bn_after_gcnn[i](x)
            l1 = self.ff_liner_layers1[i](x)
            l1 = torch.relu(l1)
            l2 = self.ff_liner_layers2[i](l1)
            x = self.bn_after_ff[i](l2)

        return x


class RCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.rnn = nn.RNN(self.hidden_size, self.hidden_size)
        self.cnn = GatedCNN(config)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.cnn(x)
        return x


class BertLSTM(nn.Module):
    def __int__(self, config,bert_config):
        super().__int__()
        # self.bert = BertModel(config["pretrain_model_path"], return_dict=False)
        self.bert = BertModel(bert_config)
        self.rnn = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, batch_first=True)

    def forward(self, x):
        x = self.bert(x)[0]
        x, _ = self.rnn(x)
        return x


class BertCNN(nn.Module):
    def __init__(self, config,bert_config):
        super().__init__()
        # self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        self.bert = BertModel(bert_config)
        config["hidden_size"] = self.bert.config.hidden_size
        self.cnn = CNN(config)

    def forward(self, x):
        x = self.bert(x)[0]
        x = self.cnn(x)
        return x


class BertMidLayer(nn.Module):
    def __init__(self, config,bert_config):
        super().__init__()
        # self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        self.bert = BertModel(bert_config)
        self.bert.output_hidden_states = True

    def forward(self, x):
        layer_states = self.bert(x)[2]
        layer_states = torch.add(layer_states[-2], layer_states[-1])
        return layer_states


def choose_optimizer(config, model):
    # print(config["optimizer"])
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
    elif optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)


if __name__ == '__main__':
    from config import Config

    Config["model_type"] = "bert"
    model = BertModel.from_pretrained(Config["pretrain_model_path"], return_dict=False)
    x = torch.LongTensor([[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]])
    print(model(x))
