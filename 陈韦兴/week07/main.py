from config import Config
from loader import load_data
from model import TorchModel, choose_optimizer
import numpy as np
from evaluate import Evaluate
import csv


def main(config):
    train_data_list = load_data(config["train_data_path"], config, data_type="train")
    model = TorchModel(config)
    optimizer = choose_optimizer(config, model)
    evaluator = Evaluate(config, model)

    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        print(f"第{epoch}轮训练")
        train_loss = []
        for index, batch_data in enumerate(train_data_list):
            optimizer.zero_grad()
            input_ids, labels = batch_data
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()
            # print(f"本轮loss为{loss.item()}")

            train_loss.append(loss.item())
        print(f"第{epoch}轮loss值为：{np.mean(train_loss)}")
        acc, time = evaluator.eval()
        print(f"第{epoch}轮正确率为：{acc}")
        writeCsv(config, acc=acc, time=time, loss=np.mean(train_loss), epoch=epoch)

    return acc


def writeCsv(config, loss=0, acc=0, time=0, epoch=0):
    with open("output.csv", "a", encoding="utf8", newline="") as f:
        header = ["model", "learning_rate", "hidden_size", "batch_size","pooling_style", "epoch", "valid_count", "loss","acc", "time"]
        writer = csv.DictWriter(f, fieldnames=header)
        # writer.writeheader()
        row_data = {
            "model": config["model_type"],
            "learning_rate": config["learning_rate"],
            "hidden_size": config["hidden_size"],
            "batch_size": config["batch_size"],
            "pooling_style": config["pooling_style"],
            "epoch": epoch,
            "valid_count": config["valid_count"],
            "loss": loss,
            "acc": acc,
            "time": time
        }
        writer.writerow(row_data)


if __name__ == '__main__':
    # main(Config)
    for model in ['lstm']:
        Config["model_type"] = model
        for lr in [1e-3, 1e-4]:
            Config["learning_rate"] = lr
            for hidden_size in [128]:
                Config["hidden_size"] = hidden_size
                for batch_size in [64, 128]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg", 'max']:
                        Config["pooling_style"] = pooling_style
                        print("最后一轮准确率：", main(Config), "当前配置：", Config)
