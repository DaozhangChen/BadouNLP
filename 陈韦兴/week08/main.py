import torch
from loader import load_data
from model import Calc_Vector_Loss,choose_optimizer
from config import Config
import numpy as np
from evaluate import Evaluate

def main(config):
    train_data = load_data(config,"train")
    model = Calc_Vector_Loss(config)
    optimizer = choose_optimizer(config,model)

    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        model = model.cuda()
    
    for epoch in range(config["epoch"]):
        epoch += 1
        train_loss = []
        print(f"开始第{epoch}轮训练")
        for index,batch_data in enumerate(train_data):
            model.train()
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            anchor,positive,negative = batch_data
            loss = model(anchor,positive,negative)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        
        print(f"本轮loss值为：{np.mean(train_loss)}")
        Evaluate(config,model)
    torch.save(model.state_dict(),f"./output/model.pth")
    
    return
            
    
    

if __name__ == "__main__":
    main(Config)