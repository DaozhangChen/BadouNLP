from config import Config
from model import Model,choose_optim
from loader import get_data_loader
from evaluate import Evaluate
import numpy as np
import torch

def main(config):
    cuda_flag = torch.cuda.is_available()
    model = Model(config)
    model.load_state_dict(torch.load("./output/model_weight2.pth",weights_only=True))
    if cuda_flag:
        model = model.cuda()
    train_data = get_data_loader(config,"train")
    optimizer = choose_optim(config,model)
    evaluate = Evaluate(config,model)


    for epoch in range(config["epoch"]):
        model.train()
        epoch += 1
        loss_items = []
        print(f"第{epoch}轮训练开始：")
        for batch_data in train_data:
            if cuda_flag :
                batch_data = [d.cuda() for d in batch_data]
            data,target,mask = batch_data
            optimizer.zero_grad()
            loss = model(data,target,attention_mask=mask)
            loss.backward()
            optimizer.step()
            loss_items.append(loss.item())
            
            # print(f"本轮loss为{loss}")
        evaluate.eval()
        print(f"第{epoch}轮loss值为{np.mean(loss_items)}")
        print("------------------------\n")
        torch.save(model.state_dict(),"./output/model_weight2.pth")
    
    
        

        


    



if __name__ == "__main__":
    main(Config)