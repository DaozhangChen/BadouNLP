from loader import get_data_loader
import re
import torch
from collections import defaultdict
import numpy as np
class Evaluate:
    def __init__(self,config,model):
        self.config = config
        self.model = model
        self.valid_data = get_data_loader(config,"valid",shuffle=False)
        self.cuda_flag = torch.cuda.is_available()
        self.sentence_list = self.valid_data.dataset.sentence
        
        



    def eval(self):
        self.stats = {
            "LOCATION":defaultdict(int),
            "ORGANIZATION":defaultdict(int),
            "PERSON":defaultdict(int),
            "TIME":defaultdict(int)
        }
        self.model.eval()
        for index,data in enumerate(self.valid_data):
            if self.cuda_flag :
                data = [d.cuda() for d in data]
            data_ids,target,mask = data
            target_sentence = self.sentence_list[index*self.config["batch_size"]:(index+1)*self.config["batch_size"]]
            with torch.no_grad():
                predict = self.model(data_ids,attention_mask=mask)
                # maxarg_list = torch.argmax(predict,dim=-1)
                # maxarg_list = predict.cpu().detach().tolist()
                maxarg_list = predict
                target = target.cpu().tolist()
                for true_target,pre_target,sentence in zip(target,maxarg_list,target_sentence):
                    true_text_dict = self.match_text(true_target,sentence)
                    pre_text_dict = self.match_text(pre_target,sentence)
                    self.write_stats(true_text_dict,pre_text_dict)
        self.show_stats()


                    
                
    def match_text(self,true_target,sentence):
        text_dict = {
            "LOCATION":[],
            "ORGANIZATION":[],
            "PERSON":[],
            "TIME":[]
        }
        true_target_text = "".join([str(num) for num in true_target if num != -1])
        for location in re.finditer("(04+)",true_target_text):
            s,e = location.span()
            text_dict["LOCATION"].append(sentence[s-1:e-1])
        for location in re.finditer("(15+)",true_target_text):
            s,e = location.span()
            text_dict["ORGANIZATION"].append(sentence[s-1:e-1])
        for location in re.finditer("(26+)",true_target_text):
            s,e = location.span()
            text_dict["PERSON"].append(sentence[s-1:e-1])
        for location in re.finditer("(37+)",true_target_text):
            s,e = location.span()
            text_dict["TIME"].append(sentence[s-1:e-1])

        return text_dict
    
    def write_stats(self,true_dict,predict_dict):
        for key in ["LOCATION","ORGANIZATION","PERSON","TIME"]:
            self.stats[key]["正确识别"] += len([value for value in predict_dict[key] if value in true_dict[key]])
            self.stats[key]["样本实体数"] += len(true_dict[key])
            self.stats[key]["预测实体数"] += len(predict_dict[key])



    def show_stats(self):
        f1_scores = []
        for key in ["LOCATION","ORGANIZATION","PERSON","TIME"]:
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            accuracy = self.stats[key]["正确识别"] / (self.stats[key]["预测实体数"] + 1e-5)
            recall = self.stats[key]["正确识别"] / (self.stats[key]["样本实体数"] +1e-5)
            F1 = count_f1(accuracy,recall)
            f1_scores.append(F1)
            print(f"{key}类实体，正确率：{accuracy:.2%}，召回率：{recall:.2%}，F1值：{F1:.2%}")
        print(f"Macro-F1值为：{np.mean(f1_scores):.2%}")
        total_correct = sum([self.stats[key]["正确识别"] for key in ["LOCATION","ORGANIZATION","PERSON","TIME"]])
        total_predict_enti = sum([self.stats[key]["预测实体数"] for key in ["LOCATION","ORGANIZATION","PERSON","TIME"]])
        total_target_enti = sum([self.stats[key]["样本实体数"] for key in ["LOCATION","ORGANIZATION","PERSON","TIME"]])
        total_accuracy = total_correct / (total_predict_enti + 1e-5)
        total_recall = total_correct / (total_target_enti + 1e-5)
        micro_f1 = count_f1(total_accuracy,total_recall)
        print(f"Micro-F1值为{micro_f1:.2%}")




def count_f1(acc,recall):
    return (2*acc*recall)/(acc + recall + 1e-5)


        


