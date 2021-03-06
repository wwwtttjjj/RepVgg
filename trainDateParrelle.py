import torch
import torchvision
import numpy as np
import os
import model
import wandb
import argparse
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

parse = argparse.ArgumentParser(description='input parameter of trian')

parse.add_argument('--batch-size', help = 'the size of training datesets', default=64,type=int)
parse.add_argument('--lr', help = 'lr', default=1e-3,type=float)
parse.add_argument('--epochs', help = 'nums of train', default=50)
parse.add_argument('--wandb', help = 'use or not use wandb', default=True, type=bool)
# parse.add_argument('--yaml', help = 'yaml file path, edit', type = str)
args = parse.parse_args()

#init more gpus

local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ["WORLD_SIZE"])
torch.distributed.init_process_group(backend='nccl', rank = local_rank, world_size = world_size)
device = torch.device('cuda', local_rank)

# device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
'''
data
'''
train_root = "./data/cif10/train/raw-img"
test_root = "./data/cif10/test/raw-img"
transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean = [0.485, 0.456,0.406],std = [0.229, 0.224, 0.225])
])
data_iter = torchvision.datasets.ImageFolder(root=train_root, transform=transform)
data_load = torch.utils.data.DataLoader(data_iter, batch_size = args.batch_size, shuffle = False, sampler = DistributedSampler(data_iter))

test_iter = torchvision.datasets.ImageFolder(root=test_root, transform=transform)
test_load = torch.utils.data.DataLoader(test_iter, batch_size = args.batch_size, shuffle = False, sampler = DistributedSampler(test_iter))

lossf = torch.nn.CrossEntropyLoss()
if args.wandb and local_rank == 0:
    wandb.init(entity='wtj', project='Repvgg object classifier')
    wandb.config = {
        'lr':args.lr,
        'epochs':args.epochs, 
        'batch_size':args.batch_size
    }

if os.path.exists('./RepVgg.pt'):
    modelRepVgg = torch.load('./RepVgg.pt', map_location=torch.device(device))
else:
    modelRepVgg = model.get_RepVgg_func_by_name('RepVgg-A0')(num_classes = 10).to(device)

if torch.cuda.device_count() > 1:
    print('use',torch.cuda.device_count(),'gpus')
    modelRepVgg = DDP(modelRepVgg, device_ids = [local_rank], output_device = local_rank)
modelRepVgg.train()

optimizer = torch.optim.SGD(modelRepVgg.parameters(), lr = args.lr)

for eopch in range(args.epochs):
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
        train_mean_loss = torch.tensor(loss_arr).mean()
        correct = 0
        total = 0
        for x, y in test_load:
            x, y = x.to(device),y.to(device)
            y_p = modelRepVgg(x)
            y_p = y_p.cpu()
            y_p = torch.argmax(y_p, dim = 1)
            label = y.cpu()
            correct +=  (y_p == label).sum()
            total += args.batch_size
        test_acc = correct / total
        print(local_rank,"test acc is",test_acc.item(), "loss is",train_mean_loss.item())
        if args.wandb and local_rank == 0:
            wandb.log({'train_mean_loss':train_mean_loss,'test_acc:':test_acc})    
    modelRepVgg.train()
#??????????????????????????????????????????????????????
    torch.save(modelRepVgg.module, './RepVgg.pt')
    torch.save(model.repVgg_model_convert(modelRepVgg.module), './RepVgg_Convert.pt')

    


