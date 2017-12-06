# Code influenced by: https://github.com/pytorch/examples/blob/master/mnist/main.py

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
# import matplotlib.pyplot as plt


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
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

no_cuda=False

cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

if cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class MLPNetModified(nn.Module):
    def __init__(self, f11, f12, f21, f22, f31,f32):
        super(MLPNetModified, self).__init__()
        self.f11 = f11
        self.f12 = f12
        self.f21 = f21
        self.f22 = f22
        self.f31 = f31
        self.f32 = f32
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
        # self.ceriation = nn.CrossEntropyLoss()
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        half = int(len(x[0])/2)
        first_part = x[:, 0:half]
        second_part = x[:, half:]
        first_part = self.f11(first_part)
        second_part = self.f12(second_part)
        x = torch.cat((first_part, second_part), 1)
        x = self.fc2(x)
        half = int(len(x[0])/2)
        first_part = x[:, 0:half]
        second_part = x[:, half:]
        first_part = self.f21(first_part)
        second_part = self.f22(second_part)
        x = torch.cat((first_part, second_part), 1)
        x = self.fc3(x)
        half = int(len(x[0])/2)
        first_part = x[:, 0:half]
        second_part = x[:, half:]
        first_part = self.f31(first_part)
        second_part = self.f32(second_part)
        x = torch.cat((first_part, second_part), 1)
        # loss = self.ceriation(x, target)
        return F.log_softmax(x)
    def name(self):
        return 'mlpnet'

plots_test_loss = []
plots_train_loss = []
plots_test_accuracy = []

def solve(f11,f12, f21,f22, f31,f32):
    # print("| "+str(f1).split()[1]+"\t| "+str(f2).split()[1]+"\t| "+str(f3).split()[1]+"\t|")
    model = MLPNetModified(f11, f12, f21, f22, f31,f32)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
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
            if batch_idx % args.logInterval == 0:
                train_loss.append(loss.data[0])
                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, batch_idx * len(data), len(train_loader.dataset),
                #     100. * batch_idx / len(train_loader), loss.data[0]))
        # print (epoch, loss_to_print)
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
            File.write('| '+str(f11).split()[1]+','+str(f12).split()[1]+'\t| '+str(f21).split()[1]+','+str(f22).split()[1]+'\t| '+str(f31).split()[1]+','+str(f32).split()[1]+'\t| '+'{:.2f}%\t|\n'.format(100. * correct / len(test_loader.dataset)))
        test_losses.append(test_loss)
        test_accuracy.append(100. * correct / len(test_loader.dataset))
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)

fileName="Results.md"
x=0
for count in range(1,2):
    File=open(fileName,'a+')
    File.write('\n\n\n')
    File.write('## Iteration: ({:.0f}%)\n\n'.format(count))
    File.write('| Layer 1\t\t| Layer 2\t\t| Layer 3\t\t| Accuracy\t\t|\n')
    File.write('| ------------- |:-------------:|:-----:| -----:|\n')
    File.close()
    a_list=[F.relu, F.tanh, F.sigmoid, F.tanh]
    c_list=[F.relu, F.tanh, F.sigmoid, F.tanh]
    e_list=[F.relu, F.tanh, F.sigmoid, F.tanh]
    for a in [F.sigmoid]:
        if(a==F.relu):
            b_list=[F.relu,F.tanh, F.sigmoid, F.selu]
        elif(a==F.tanh):
            b_list=[F.tanh,F.sigmoid, F.selu]
        elif(a==F.sigmoid):
            b_list=[F.sigmoid,F.selu]
        else:
            b_list=[F.selu]
        for b in [F.sigmoid]:
            for c in [F.sigmoid]:
                if(c==F.relu):
                    d_list=[F.relu,F.tanh, F.sigmoid, F.selu]
                elif(c==F.tanh):
                    d_list=[F.tanh,F.sigmoid, F.selu]
                elif(c==F.sigmoid):
                    d_list=[F.sigmoid,F.selu]
                else:
                    d_list=[F.selu]
                for d in [F.sigmoid]:
                    for e in e_list:
                        if(e==F.relu):
                            f_list=[F.relu,F.tanh, F.sigmoid, F.selu]
                        elif(e==F.tanh):
                            f_list=[F.tanh,F.sigmoid, F.selu]
                        elif(e==F.sigmoid):
                            f_list=[F.sigmoid,F.selu]
                        else:
                            f_list=[F.selu]
                        for f in f_list:
                            x+=1
                            print(x)
                            print(a, b, c, d, e, f)
                            File=open(fileName,'a+')
                            solve(a, b, c, d, e, f)
                            File.close()

# File=open(fileName,'a+')
# solve(F.sigmoid, F.sigmoid, F.sigmoid, F.sigmoid,F.selu, F.selu)
# solve(F.sigmoid, F.sigmoid, F.sigmoid, F.sigmoid,F.selu, F.selu)
# File.close()
