from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt


batch_size = 32
test_batch_size = 1000
epochs = 10
lr = 0.01
momentum = 0.5
no_cuda = False
seed = 1
log_interval = 10

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
        # self.ceriation = nn.CrossEntropyLoss()
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        half = int(len(x[0])/2)
        first_part = x[:, 0:half]
        second_part = x[:, half:]
        first_part = self.f1(first_part)
        second_part = self.f1(second_part)
        x = torch.cat((first_part, second_part), 1)
        x = self.fc2(x)
        half = int(len(x[0])/2)
        first_part = x[:, 0:half]
        second_part = x[:, half:]
        first_part = self.f2(first_part)
        second_part = self.f2(second_part)
        x = torch.cat((first_part, second_part), 1)
        x = self.fc3(x)
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

plots_test_loss = []
plots_train_loss = []
plots_test_accuracy = []

def solve(f1, f2, f3, _optimizer= "sgd"):
    # print (str(f1).split()[1], str(f2).split()[1], str(f3).split()[1], _optimizer)
    model = MLPNetModified(f1, f2, f3)
    if(_optimizer == "sgd"):
        optimizer = optim.SGD(model.parameters(), lr = lr, momentum = momentum)
    elif(_optimizer == "rmsprop"):
        optimizer = optim.RMSprop(model.parameters(), lr = lr)
    elif(_optimizer == "adagrad"):
        optimizer = optim.Adagrad(model.parameters(), lr = lr)
    elif(_optimizer == "adadelta"):
        optimizer = optim.Adadelta(model.parameters(), lr = lr)
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
            loss_to_print += loss.data[0]
            if batch_idx % log_interval == 0:
                train_loss.append(loss.data[0])
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
        test_losses.append(test_loss)
        test_accuracy.append(100. * correct / len(test_loader.dataset))
        if (epoch == epochs):
            return 100. * correct / len(test_loader.dataset)
        return 0
    for epoch in range(1, epochs + 1):
        train(epoch)
        accuracy = test(epoch)
    return accuracy
index=1
print("| Sl.No | Layers | SGD | RMSprop | Adadelta | Adagrad |")
print("|-------|---------|------|----------|-----------|----------|")
for a in [F.relu, F.tanh, F.sigmoid, F.selu]:
    for b in [F.relu, F.tanh, F.sigmoid, F.selu]:
        for c in [F.relu, F.tanh, F.sigmoid, F.selu]:
            res1 = solve(a, b, c, "sgd")
            # print("one")
            res2 = solve(a, b, c, "rmsprop")
            # print("Two")
            res3 = solve(a, b, c, "adadelta")
            # print("Three")
            res4 = solve(a, b, c, "adagrad")
            # print("Four")
            print("| "+str(index)+"\t| "+str(res1)+"\t| "+str(res2)+"\t| "+str(res3)+"\t| "+str(res4)+"\t|")
