from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
# import matplotlib.pyplot as plt


batch_size = 32
test_batch_size = 1000
epochs = 10
lr = 0.01
momentum = 0.5
no_cuda = False
seed = 1
log_interval = 10

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--logInterval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)

if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=True, **kwargs)

class MLPNetModified(nn.Module):
    def __init__(self, f1, f2, f3):
        super(MLPNetModified, self).__init__()
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
        self.bn1 = nn.BatchNorm1d(500)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(10)
        # self.ceriation = nn.CrossEntropyLoss()
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.bn1(self.fc1(x))
        half = int(len(x[0])/2)
        first_part = x[:, 0:half]
        second_part = x[:, half:]
        first_part = self.f1(first_part)
        second_part = self.f1(second_part)
        x = torch.cat((first_part, second_part), 1)
        x = self.bn2(self.fc2(x))
        half = int(len(x[0])/2)
        first_part = x[:, 0:half]
        second_part = x[:, half:]
        first_part = self.f2(first_part)
        second_part = self.f2(second_part)
        x = torch.cat((first_part, second_part), 1)
        x = self.bn3(self.fc3(x))
        half = int(len(x[0])/2)
        first_part = x[:, 0:half]
        second_part = x[:, half:]
        first_part = self.f3(first_part)
        second_part = self.f3(second_part)
        x = torch.cat((first_part, second_part), 1)
        # loss = self.ceriation(x, target)
        return F.log_softmax(x)
    def name(self):
        return 'mlpnet'

def solve(f1, f2, f3, count):
    print (str(f1).split()[1], str(f2).split()[1], str(f3).split()[1])
    if(cuda):
        model = MLPNetModified(f1, f2, f3).cuda()
    else:
        model = MLPNetModified(f1, f2, f3)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    train_loss = []
    test_losses = []
    test_accuracy = []
    def train(epoch):
        model.train()
        loss_to_print = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            loss_to_print = loss.data[0]
            if batch_idx % log_interval == 0:
                train_loss.append(loss.data[0])
        print (epoch, loss_to_print)
    def test(epoch):
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(test_loader.dataset)
        if (epoch == args.epochs):
            index=0
            dic={"relu":0,"tanh":1,"sigmoid":2,"selu":3}
            F1,F2,F3=str(f1).split()[1],str(f2).split()[1],str(f3).split()[1]
            # print(F1,F2,F3)
            index+=dic[F1]*16+dic[F2]*4+dic[F3]
            acc_arr[index][count]=round(100. * correct / len(test_loader.dataset),2)
            print("ACCURACY:")
            print(acc_arr[index][count])
            # print(acc_arr)
#             File.write('| '++'\t| '+str(f2).split()[1]+'\t| '+str(f3).split()[1]+'\t| '+'{:.2f}%\t|\n'.format(100. * correct / len(test_loader.dataset)))
        test_losses.append(test_loss)
        test_accuracy.append(100. * correct / len(test_loader.dataset))
    for epoch in range(1, epochs + 1):
        train(epoch)
        test(epoch)
FileName="BatchNormResults.md"
File=open(FileName,'a+')
File.write('\n\n\n')
File.write('| Layer 1   | Layer 2   | Layer 3   | Iter 1    |Iter 2 |Iter 3 |Iter 4 |Iter 5 |Iter 6 |Iter 7 |Iter 8 |Iter 9 | Iter10 | mean  |std.dev|\n')
File.write('| ------- |:-------:|:-------:|:-------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:| -----:|\n')
File.close()
x=0
acc_arr=np.zeros((64,10))
for count in range(0,10):
    for a in [F.relu, F.tanh, F.sigmoid, F.selu]:
        for b in [F.relu, F.tanh, F.sigmoid, F.selu]:
            for c in [F.relu, F.tanh, F.sigmoid, F.selu]:
                x+=1
                print(x)
                print(a, b, c)
                solve(F.selu, F.selu,F.selu,0)
                solve(F.selu, F.selu,F.sigmoid,0)
                solve(F.selu, F.selu,F.selu,0)
                solve(F.selu, F.selu,F.sigmoid,0)
                solve(F.selu, F.selu,F.selu,0)
                solve(F.selu, F.selu,F.sigmoid,0)
                solve(F.selu, F.selu,F.selu,0)
                solve(F.selu, F.selu,F.sigmoid,0)

File=open(FileName,'a+')
dic1={0:"relu",1:"tanh",2:"sigmoid",3:"selu"}
for i in range(0,len(acc_arr)):
    File.write("| ")
    l=int(i/16)
    m=int((i-l*16)/4)
    n=int(i-l*16-m*4)
    print(l,m,n)
    File.write(dic1[l]+"\t| ")
    File.write(dic1[m]+"\t| ")
    File.write(dic1[n]+"\t| ")
    for j in range(0,len(acc_arr[0])):
        File.write(str(acc_arr[i][j])+"\t| ")
    File.write(str(round(np.average(acc_arr[i]),2))+"\t| "+str(round(np.std(acc_arr[i]),2))+"\t|")
    File.write("\n")
File.close()




