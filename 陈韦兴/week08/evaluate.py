from loader import DataGenerator
from config import Config
import torch
from model import Calc_Vector_Loss

class Evaluate:
    def __init__(self,config,model):
        self.cuda_flag = torch.cuda.is_available()
        self.config = config
        self.model = model
        self.valid_dg = DataGenerator(self.config,"valid")
        self.train_dg = DataGenerator(self.config,"train")
        self.train_data_dict = self.train_dg.get_vector_question_dict()
        self.valid_data_dict = self.valid_dg.get_vector_question_dict()
        self.stats = {"correct":0,"wrong":0}
        
        self.eval()


    def eval(self):
        self.model.eval()
        self.set_all_question_vector()
        with torch.no_grad():
            for type,questions in self.valid_data_dict.items():
                for question in questions:
                    if self.cuda_flag:
                        question_vector = torch.LongTensor([question]).cuda()
                    predict = self.model(question_vector)
                    predict = torch.nn.functional.normalize(predict,dim=-1)
                    maxarg = int(find_maxarg(self.all_question_vector_dict,predict))
                    if self.all_question_type_dict[maxarg] == type:
                        self.stats["correct"] += 1
                    else:
                        self.stats["wrong"] += 1

        print(f"本轮正确率为：{(self.stats["correct"]/(self.stats["correct"]+self.stats["wrong"])):.2%}")
        return
        


    def set_all_question_vector(self):
        with torch.no_grad():
            self.all_question_type_dict = []
            self.all_question_vector_dict = []
            for type,questions in self.train_data_dict.items():
                # question = torch.LongTensor(question)
                for question in questions:
                    self.all_question_type_dict.append(type)
                    self.all_question_vector_dict.append(question)
            if self.cuda_flag:
                self.all_question_vector_dict = torch.LongTensor(self.all_question_vector_dict).cuda()
            self.all_question_vector_dict = self.model(self.all_question_vector_dict)
            # 一定要加上正则化，否则正确率会越来越低
            self.all_question_vector_dict = torch.nn.functional.normalize(self.all_question_vector_dict,dim=-1)


def find_maxarg(all_question_vector,current_question_vector):
    calc_vector = torch.mm(current_question_vector.unsqueeze(0),all_question_vector.T)
    return torch.argmax(calc_vector)


    

    
        






if __name__ == "__main__":
    Config["vocab_size"] = 14000
    model = Calc_Vector_Loss(Config)
    evaluate = Evaluate(Config,model)
        