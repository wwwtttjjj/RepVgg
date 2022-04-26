
import torch
import torchvision
import numpy as np
import os
import model
import wandb

batch_size = 8
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
'''
data
'''
train_root = "D:/deep_learing_work/data/cif10/train/raw-img"
test_root = "D:/deep_learing_work/data/cif10/test/raw-img"
transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean = [0.485, 0.456,0.406],std = [0.229, 0.224, 0.225])
])
data_iter = torchvision.datasets.ImageFolder(root=train_root, transform=transform)
data_load = torch.utils.data.DataLoader(data_iter, batch_size = batch_size, shuffle = True)

test_iter = torchvision.datasets.ImageFolder(root=test_root, transform=transform)
test_load = torch.utils.data.DataLoader(test_iter, batch_size = batch_size, shuffle = False)

epochs = 20
lossf = torch.nn.CrossEntropyLoss()
lr = 1e-5
# wandb.init(entity='wtj', project='Repvgg object classifier')
# wandb.config = {
#     'lr':0.0001,
#     'epochs':20, 
#     'batch_size':8
# }

modelRepVgg = model.get_RepVgg_func_by_name('RepVgg-A0')(num_classes = 10).to(device)
modelRepVgg.train()
optimizer = torch.optim.SGD(modelRepVgg.parameters(), lr = lr)

for eopch in range(epochs):
    loss_arr = []
    for x_cpu, y_cpu in data_load:
        optimizer.zero_grad()
        x = x_cpu.to(device)
        y = y_cpu.to(device)
        y_p = modelRepVgg(x)
        train_loss = lossf(y_p, y).sum()
        train_loss.backward()
        loss_arr.append(train_loss.detach().cpu())
        optimizer.step()
    modelRepVgg.eval()
    with torch.no_grad():
        train_mean_loss = torch.tensor(loss_arr)
        correct = 0
        total = 0
        for x, y in test_load:
            x, y = x.to(device),y.to(device)
            y_p = modelRepVgg(x)
            y_p = y_p.cpu()
            y_p = torch.argmax(y_p, dim = 1)
            label = y.cpu()
            correct +=  (y_p == label).sum()
            total += batch_size
    print(correct / total)
    
    modelRepVgg.train()
    model.save(modelRepVgg, './RepVgg.pt')
    

