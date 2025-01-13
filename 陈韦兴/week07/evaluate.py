import time

import torch

from loader import load_data


class Evaluate:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.valid_data = load_data(self.config["train_data_path"], self.config, data_type="valid")
        self.stats_dict = {"correct": 0, "wrong": 0}

    def eval(self):
        start_time = time.time()
        self.model.eval()
        self.stats_dict = {"correct": 0, "wrong": 0}
        for index, batch_data in enumerate(self.valid_data):
            with torch.no_grad():
                input_id, label = batch_data
                output = self.model(input_id)
            self.write_stats(label, output)
        acc = self.show_stats()
        end_time = time.time()
        elapsed_time = end_time - start_time
        return acc,elapsed_time

    def write_stats(self, label, pred_result):
        for true_label, pred_result in zip(label, pred_result):
            if int(true_label) == torch.argmax(pred_result):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.config["valid_count"] = correct + wrong
        print(f"总预测个数为{correct + wrong}")
        print(f"正确数量为：{correct}")
        print(f"错误数量为：{wrong}")
        print(f"正确率为：{(correct / (correct + wrong)) * 100}%")
        print("------------------------")
        return (correct / (correct + wrong)) * 100
