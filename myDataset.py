from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import time

# from models import *
from torchvision import models
from models.resnet import ResNet18
from models.vit import ViT
from utils import progress_bar
from models.convmixer import ConvMixer
from randomaug import RandAugment
from dataset import CRC_DataSet

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')  # resnets.. 1e-3, Vit..1e-4?
parser.add_argument('--opt', default="adam")
parser.add_argument('--aug', action='store_true', help='use randomaug')
parser.add_argument('--amp', action='store_true', help='enable AMP training')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--bs', default='256')
parser.add_argument('--n_epochs', type=int, default='50')
parser.add_argument('--patch', default='4', type=int)
parser.add_argument('--convkernel', default='8', type=int)
parser.add_argument('--cos', action='store_false', help='Train with cosine annealing scheduling')


import sys
args = parser.parse_args()
def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

NUM_CLASSES = 2
classes = ('MSIMUT', 'MSS')

watermark = "{}_lr{}".format(args.net, args.lr)
if args.amp:
    watermark += "_useamp"

if args.aug:
    import albumentations
bs = int(args.bs)

use_amp = args.amp

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
# if args.net == "vit_timm":
#     size = 384
# else:
#     size = 32
size = 128
transform_train = transforms.Compose([
    transforms.RandomCrop(224, padding=18),
    transforms.Resize(size),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Add RandAugment with N, M(hyperparameter)
if args.aug:
    N = 2;
    M = 14;
    transform_train.transforms.insert(0, RandAugment(N, M))

#     ===============DATASET
from urllib import request
import os
from glob import glob
import zipfile
data_name = ['CRC_DX_TRAIN_MSIMUT', 'CRC_DX_TRAIN_MSS', 'CRC_DX_TEST_MSS', 'CRC_DX_TEST_MSIMUT']
front_link = 'https://zenodo.org/record/2530835/files/'
back_link = '.zip?download=1'

# # 폴더 디렉토리 생성
# if not os.path.isdir('TCGA_DATA'):
#     os.makedirs('TCGA_DATA')
#     os.makedirs('TCGA_DATA/CRC_TRAIN')
#     os.makedirs('TCGA_DATA/CRC_TEST')
#
#     for idx, data_type in enumerate(data_name):
#         # 코랩 아니면 아래 경로 수정
#         print(data_type+" download start")
#         if idx <= 1:
#             os.chdir('TCGA_DATA/CRC_TRAIN')
#         else:
#             os.chdir('TCGA_DATA/CRC_TEST')
#
#         link = front_link + data_type + back_link
#         request.urlretrieve(link, data_type, reporthook)
#         zipfile.ZipFile(data_type).extractall()
#         print('One Done')



DATA_PATH_TRAIN_LIST = glob('TCGA_DATA/CRC_TRAIN/*/*.png')
DATA_PATH_TEST_LIST = glob('TCGA_DATA/CRC_TEST/*/*.png')
trainloader = torch.utils.data.DataLoader(
    CRC_DataSet(
        DATA_PATH_TRAIN_LIST,
        classes,
        transform=transform_train
    ),
    batch_size=bs,
    shuffle = True
)
testloader = torch.utils.data.DataLoader(
    CRC_DataSet(
        DATA_PATH_TEST_LIST,
        classes,
        transform=transform_test
    ),
    batch_size=bs,
    shuffle = True
)

# print(trainlo)
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
if args.net == 'res18':
    net = models.resnet18(pretrained=False)
    num_classes = NUM_CLASSES
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, num_classes)
# elif args.net=='vgg':
#     net = VGG('VGG19')
# elif args.net=='res34':
#     net = ResNet34()
# elif args.net=='res50':
#     net = ResNet50()
# elif args.net=='res101':
#     net = ResNet101()
elif args.net == "convmixer":
    # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
    net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=NUM_CLASSES)
elif args.net == "vit":
    print('VIT MODEL')
    # ViT for cifar10
    net = ViT(
        image_size=size,
        patch_size=args.patch,
        num_classes=NUM_CLASSES,
        dim=64,
        depth=6,
        heads=8,
        mlp_dim=64,
        dropout=0.1,
        emb_dropout=0.1
    )
elif args.net == "vit_timm":
    import timm

    net = timm.create_model("vit_large_patch16_384", pretrained=False)
    net.head = nn.Linear(net.head.in_features, NUM_CLASSES)
    print("model adaptation finished")

elif args.net == "glasses":
    from glasses.models import AutoModel
    # from glasses.models import AutoTransform

    net = AutoModel.from_pretrained("vit_small_patch16_224")
    # cfg = AutoTransform.from_name('vit_large_patch16_224')
    # why there are no class named AutoTransform?? // Jh
    # print(cfg)
elif args.net == "This":
    # https: // github.com / lukemelas / PyTorch - Pretrained - ViT
    from pytorch_pretrained_vit import ViT
    net = ViT('L_32_imagenet1k', pretrained=True, image_size=size, num_classes=2)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)  # make parallel
    cudnn.benchmark = True
#
# if args.resume:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
#     net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']

# Loss is CE
criterion = nn.CrossEntropyLoss()

# if args.opt == "adam":
optimizer = optim.Adam(net.parameters(), lr=args.lr)
# elif args.opt == "sgd":
#     optimizer = optim.SGD(net.parameters(), lr=args.lr)

# use cosine or reduce LR on Plateau scheduling
if not args.cos:
    from torch.optim import lr_scheduler

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, min_lr=1e-3 * 1e-5,
                                               factor=0.1)
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)
#
# if args.cos:
#     wandb.config.scheduler = "cosine"
# else:
#     wandb.config.scheduler = "ReduceLROnPlateau"

##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # print(predicted, targets)
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return train_loss / (batch_idx + 1)


##### Validation
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            # print(predicted, targets)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Update scheduler
    if not args.cos:
        scheduler.step(test_loss)

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {"model": net.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict()}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + args.net + '-{}-ckpt.t7'.format(args.patch))
        best_acc = acc

    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc


list_loss = []
list_acc = []

# wandb.watch(net)
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    val_loss, acc = test(epoch)

    if args.cos:
        scheduler.step(epoch - 1)

    list_loss.append(val_loss)
    list_acc.append(acc)
    print('epoch ', epoch, ' train loss ', trainloss, ' val loss ', val_loss, ' val_acc ', acc,
          ' lr ' , optimizer.param_groups[0]["lr"], " epoch_time ",  time.time() - start)
    # Log training..
    # wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc,
    #            "lr": optimizer.param_groups[0]["lr"],
    #            "epoch_time": time.time() - start})

    # Write out csv..
    with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss)
        writer.writerow(list_acc)
    print(list_loss)

# writeout wandb
# wandb.save("wandb_{}.h5".format(args.net))

torch.save("model_{}".format(args.net))