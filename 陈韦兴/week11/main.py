from config import Config
from loader import get_data
from model import LanguageModel,choose_optimizer
import torch
import numpy as np
from evaluate import Evaluate

def main(config):
    epoch_number = config["epoch"]
    stf_data = get_data(config=config,train_type="sft")
    model = LanguageModel(config=config)
    optimizer = choose_optimizer(config=config,model=model,lr=config["learning_rate"])
    cuda_flag = torch.cuda.is_available()
    evaluate = Evaluate(model=model,config=config)
    model.load_state_dict(torch.load("./output/model_weights.pth",weights_only=True))
    if cuda_flag :
        model = model.cuda()
        
    
    
    for epoch in range(epoch_number):
        print(f"第{epoch + 1}轮开始训练:\n")
        model.train()
        loss_items = []
        count = 0
        epoch_data_number = config["epoch_data_number"]
        for index,data in enumerate(stf_data):
            input,target,attention_mask,loss_mask_index = data
            if cuda_flag:
                input = input.cuda()
                target = target.cuda()
                attention_mask = attention_mask.cuda()
                loss_mask_index = loss_mask_index.cuda()
            loss = model(input,y=target,mask=attention_mask,loss_mask_index=loss_mask_index)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_items.append(loss.item())
            count += config["batch_size"]
            if  count / epoch_data_number == 0.25:
                print(f"已训练25%")
            elif count / epoch_data_number == 0.5:
                print(f"已训练50%")
            elif count / epoch_data_number == 0.75:
                print(f"已训练75%")

        print(f"本轮loss值为:{np.mean(loss_items)}")
        evaluate.eval("西餐厅也应“入乡随俗”")
        evaluate.eval("下午茶•你，怕老吗？")
        torch.save(model.state_dict(),"./output/model_weights.pth")

    return



if __name__ == "__main__":
    main(Config)
