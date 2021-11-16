#Data preprocessing handled with code from https://github.com/imdeepmind/processed-imdb-wiki-dataset.git

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.parallel
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm
import torchvision.models as models
import torchvision.transforms as transforms
from dataset import IMDBWikiDataset
#from util.misc import CSVLogger
#salloc --account=PAS2056 --nodes=2 --gpus-per-node=1 srun --pty /bin/bash

data_file = '/users/PAS1906/demilt4/cse5449/lab1/meta.csv'

model_options = ['resnet18', 'resnet50', 'resnet101', 'wideresnet']
dataset_options = ['IMDB-Wiki']

parser = argparse.ArgumentParser(description='IMDB-Wiki Training')

parser.add_argument('--dataset', '-d', default='IMDB-Wiki',
                    choices=dataset_options)
parser.add_argument('--model', '-a', default='resnet50',
                    choices=model_options)
parser.add_argument('--num_classes','-n',type=int,default=130,
                    help = 'number of classes for age ranking')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=50,
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

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

model = models.resnet18()

in_features = model.fc.in_features
model.fc = nn.Linear(in_features, args.num_classes+1)


transform = transforms.Compose([transforms.Resize(256),
                                transforms.RandomCrop(224),
                                transforms.ToTensor()])
dataset = IMDBWikiDataset(data_file,transform)

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.67),len(dataset) - int(len(dataset) * 0.67)])

train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, pin_memory = True, num_workers = 4)
test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, pin_memory = True, num_workers = 4)

#model = nn.DataParallel(model)

print(model)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
# if not os.path.exists('logs'):
#     os.mkdir('logs')
# filename = 'logs/' + test_id + '.csv'
# csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=filename)

def test(loader):
    model.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred = model(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = (correct / total)*100
    model.train()
    return val_acc

best_accuracy = 0
best_acc_epoch = 0
epoch_progress_bar = tqdm(range(0, args.epochs),desc= 'Epoch')
for i,epoch in enumerate(epoch_progress_bar):

    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader,unit ='img',unit_scale = args.batch_size,desc = 'Mini-Batch')
    for i, (images,labels) in enumerate(progress_bar):
        #progress_bar.set_description('Epoch ' + str(epoch))

        # images = images.cuda()
        # labels = labels.cuda()

        model.zero_grad()
        pred = model(images)
        xentropy_loss = criterion(pred,labels)

        xentropy_loss.backward()
        optimizer.step()

        xentropy_loss_avg += xentropy_loss.item()

        # pred = torch.max(pred.data, 1)[1]
        # total += labels.size(0)
        # correct += (pred == labels.data).sum().item()
        # accuracy = (correct / total)*100

        # progress_bar.set_postfix(
        #     xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
        #     acc='%.3f' % accuracy)

    test_acc = test(test_loader)
    tqdm.write('test_acc: %.3f' % (test_acc))
    epoch_progress_bar.set_postfix(test_accuracy='%.3f' % test_acc)

    scheduler.step(epoch)

    #row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
    #csv_logger.writerow(row)

    if(test_acc>best_accuracy):
        best_accuracy = test_acc
        best_acc_epoch = epoch

# Create checkpoints Directory if does exist
if not os.path.exists('checkpoints'):
    os.mkdir('checkpoints')

torch.save(model.state_dict(), 'checkpoints/' + test_id + '.pt')
#csv_logger.close()

f = open("best_accuracy.txt", "a+")
f .write('best acc: %.3f at iteration: %d, and Training time %d \r\n' % (best_accuracy, best_acc_epoch))
f.close()
