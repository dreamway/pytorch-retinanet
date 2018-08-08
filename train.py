from __future__ import print_function

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from loss import FocalLoss
from retinanet import RetinaNet
from datagen import ListDataset

from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

assert torch.cuda.is_available(), 'Error: CUDA not found!'
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch
best_train_loss = float('inf')

# Data
print('==> Preparing data..')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

trainset = ListDataset(root='./data/retina-metro/',
                       list_file='./data/metro_train.txt', train=True, transform=transform, input_size=416)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2, collate_fn=trainset.collate_fn)

testset = ListDataset(root='./data/retina-metro/',
                      list_file='./data/metro_val.txt', train=False, transform=transform, input_size=416)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2, collate_fn=testset.collate_fn)

# Model
net = RetinaNet()
net.load_state_dict(torch.load('./model/net.pth'))
net.cuda()

criterion = FocalLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/ckpt_train.pth')
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optim'])  # for training, the optimizer's state_dict is also important
    print("resume best_loss:", best_loss, " resume start epoch:", start_epoch)

#net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
#net.cuda()
print(net)

# Training
def train(epoch):
    print('\nTraining, Epoch: %d' % epoch)
    net.train()
    #net.module.freeze_bn()
    net.freeze_bn()
    train_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        print('train_loss: %.3f | avg_loss: %.3f' % (loss.item(), train_loss/(batch_idx+1)))
    print("epoch train_loss: %.3f, avg_loss: %.3f"%(train_loss, train_loss/len(trainloader)))

    # state saving for resume training
    global best_train_loss
    train_loss /= len(trainloader)
    if train_loss < best_train_loss:
        print("save the training net by better train_loss...")
        state = {
            'net' : net.state_dict(),
            'loss' : train_loss,
            'epoch' : epoch,
            'optim' : optimizer.state_dict()
        }
        torch.save(state, './checkpoint/ckpt_train.pth')
        best_train_loss = train_loss

# Test
def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(testloader):
            inputs = Variable(inputs.cuda())
            loc_targets = Variable(loc_targets.cuda())
            cls_targets = Variable(cls_targets.cuda())

            loc_preds, cls_preds = net(inputs)
            loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            test_loss += loss.item()
            print('test_loss: %.3f | avg_loss: %.3f' % (loss.item(), test_loss/(batch_idx+1)))

    # Save checkpoint
    global best_loss
    test_loss /= len(testloader)
    print("epoch %d, test_loss: %.3f"%(epoch, test_loss))

    if test_loss < best_loss:
        print('Saving the net by best_loss..')
        state = {
            'net': net.state_dict(),
            'loss': test_loss,
            'epoch': epoch,            
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_loss = test_loss


for epoch in range(start_epoch, start_epoch+5000):
    train(epoch)
    test(epoch)
