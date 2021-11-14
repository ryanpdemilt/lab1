#Data preprocessing handled with code from https://github.com/imdeepmind/processed-imdb-wiki-dataset.git

import argparse
import time
import socket
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import MultiStepLR
import torch.distributed as dist
import torch.nn.parallel
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm
import torchvision.models as models
import torchvision.transforms as transforms
from dataset import IMDBWikiDataset

from torch.nn.parallel import DistributedDataParallel as DDP
#from util.misc import CSVLogger

data_file = '/users/PAS1906/demilt4/cse5449/lab1/meta.csv'

model_options = ['resnet18', 'resnet50', 'resnet101', 'wideresnet']
dataset_options = ['IMDB-Wiki']

def train(args,local_world_size,rank):

    n = torch.cuda.device_count() // local_world_size
    device_ids = list(range(rank * n, (rank + 1) * n))

    print(
        f"[{os.getpid()}] rank = {dist.get_rank()}, "
        + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {device_ids} \n", end=''
    )
    model = models.resnet18()
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, args.num_classes+1)
    
    model = DDP(model.cuda())

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    criterion = nn.CrossEntropyLoss().cuda()


    transform = transforms.Compose([transforms.Resize(256),
                                transforms.RandomCrop(224),
                                transforms.ToTensor()])
    dataset = IMDBWikiDataset(data_file,transform)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.67),len(dataset) - int(len(dataset) * 0.67)])

    train_sampler = DistributedSampler(train_dataset,shuffle = True)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size,shuffle=False, pin_memory = True,sampler = train_sampler)


    for epoch in range(0, args.epochs):
        epoch_start = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda()
            labels = labels.cuda()

            pred = model(images)


            xentropy_loss = criterion(pred,labels)

            xentropy_loss.backward()
            optimizer.step()
        epoch_end = time.time()
        diff = epoch_end-epoch_start
        throughput_local = len(train_dataset) / diff
    print(throughput_local)
    throughput_local = torch.tensor(throughput_local)
    dist.all_reduce(throughput_local)
    print('Reduced Throughput: ')
    print(throughput)
    return throughput

def ddp_main(args,local_world_size,local_rank):
    os.environ['RANK'] = os.environ['SLURM_PROCID']
    os.environ['MASTER_ADDR'] = socket.gethostbyname(os.environ['MASTER_ADDR'])
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")  
    dist.init_process_group(backend="gloo", rank = int(env_dict["RANK"]),world_size=int(env_dict["WORLD_SIZE"]))
    print(
        f"[{os.getpid()}]: world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()} \n", end=''
    )
    throughput = train(args,local_world_size,local_rank)

    #print(f'Throughput: {throughput.item()}')
    dist.destroy_process_group()

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IMDB-Wiki Training')

    parser.add_argument('--dataset', '-d', default='IMDB-Wiki',
                    choices=dataset_options)
    parser.add_argument('--model', '-a', default='resnet50',
                    choices=model_options)
    parser.add_argument('--num_classes','-n',type=int,default=130,
                    help = 'number of classes for age ranking')
    parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train (default: 120)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 1)')
    parser.add_argument('--beta', default=1, type=float,
                     help='hyperparameter beta')
    parser.add_argument('--local_rank',type=int,default=0)
    parser.add_argument('--local_world_size',type=int,default=1)
    args = parser.parse_args()
    ddp_main(args,args.local_world_size,args.local_rank)