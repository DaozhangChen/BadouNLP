from model import Calc_Vector_Loss
from config import Config
from loader import DataGenerator
import torch
from evaluate import find_maxarg
class Predict:
    def __init__(self,config):
        self.config = config
        self.cuda_flag = torch.cuda.is_available()

        self.train_data = DataGenerator(self.config,"train")
        self.train_data_dict = self.train_data.get_vector_question_dict()
        self.schema_dict = self.train_data.schema_dict

        if self.cuda_flag:
            self.model = Calc_Vector_Loss(config)
            self.model_weights = torch.load(f"./output/model.pth",map_location = "cuda:0",weights_only=True)
            self.model.load_state_dict(self.model_weights)
            self.model = self.model.cuda()
        else:
            self.model = Calc_Vector_Loss(config)
            self.model_weights = torch.load(f"./output/model.pth",map_location = "cpu",weights_only=True)
            self.model.load_state_dict(self.model_weights)
        self.set_all_question_vector()


    def predict(self):
        start_flag = True
        while start_flag:
            input_question = input("请输入问题：")
            if(input_question == "quit"):
                start_flag = False
            else:
                sentence_code = torch.LongTensor([self.train_data.format_sentence(input_question)])
                if self.cuda_flag:
                    sentence_code = sentence_code.cuda()
                predict_sentence = self.model(sentence_code)
                maxarg = find_maxarg(self.all_question_vector_dict,predict_sentence)
                question_type_number = self.all_question_type_dict[maxarg]
                question_type_name = [key for key,value in self.schema_dict.items() if value == question_type_number][0]
                print(question_type_name)






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


if __name__ == "__main__":
    predict = Predict(Config)
    
    predict.predict()
