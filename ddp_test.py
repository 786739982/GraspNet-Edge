import torch
import torch.nn as nn
from torch.optim import SGD
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

# 定义一个随机数据集
class RandomDataset(Dataset):
    def __init__(self, dataset_size, image_size=32):
        images = torch.randn(dataset_size, 3, image_size, image_size)
        labels = torch.zeros(dataset_size, dtype=int)
        self.data = list(zip(images, labels))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

# 定义模型
class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(3, 16, 3)
        self.fc = nn.Linear(30*30*16, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv2d(x)
        x = x.reshape(batch_size, -1)
        x = self.fc(x)
        out = self.softmax(x)
        return out


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default='0,1,2,3,4,5,6,7')
parser.add_argument('--batchSize', type=int, default=64)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--dataset-size', type=int, default=1024)
parser.add_argument('--num-classes', type=int, default=10)
config = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
torch.distributed.init_process_group(backend="nccl")

local_rank = torch.distributed.get_rank()
print(local_rank)
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

# 实例化模型、数据集和加载器loader
model = Model(config.num_classes)

dataset = RandomDataset(config.dataset_size)
sampler = DistributedSampler(dataset) # 这个sampler会自动分配数据到各个gpu上
loader = DataLoader(dataset, batch_size=config.batchSize, sampler=sampler)

# loader = DataLoader(dataset, batch_size=config.batchSize, shuffle=True)
loss_func = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    model.cuda()
    print('cuda is ok')
print('Preparing Distributed!')
model = torch.nn.parallel.DistributedDataParallel(model)
print('Distributed!')
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)


# 开始训练
for epoch in range(config.epochs):
    model.train()
    for step, (images, labels) in enumerate(loader):
        if torch.cuda.is_available(): 
            images = images.cuda()
            labels = labels.cuda()
        preds = model(images)
        # print(f"data: {images.device}, model: {next(model.parameters()).device}")
        loss = loss_func(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'Step: {step}, Loss: {loss.item()}')

    print(f'Epoch {epoch} Finished !')
