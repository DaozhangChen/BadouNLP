from transformers import BertTokenizer
import torch
class Evaluate:
    def __init__(self,model,config):
        self.model = model
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(self.config["bert_path"])
        

    def eval(self,input):
        with torch.no_grad():
            final_text = input + "[SEP]"
            self.model.eval()
            pred_text = ""
            while pred_text != "\n" and len(final_text)<=50:
                final_text += pred_text
                tokens = self.tokenizer.encode(final_text,add_special_tokens=False)
                tokens = torch.LongTensor([tokens])
                if torch.cuda.is_available():
                    tokens = tokens.cuda()
                pred = self.model(tokens)
                index = torch.argmax(pred)
                pred_text = self.tokenizer.decode([index])
            print(final_text)


        