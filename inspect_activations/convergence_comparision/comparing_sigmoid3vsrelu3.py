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
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
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
        return F.log_softmax(x)
    def name(self):
        return 'mlpnet'

def solve(f1, f2, f3 ,lr):
    print (str(f1).split()[1], str(f2).split()[1], str(f3).split()[1])
    model = MLPNetModified(f1, f2, f3)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    train_loss = []
    test_losses = []
    test_accuracy = []
    def train(epoch):
        model.train()
        loss_to_print = 0
        for data, target in train_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            loss_to_print += loss.data[0]
                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, batch_idx * len(data), len(train_loader.dataset),
                #     100. * batch_idx / len(train_loader), loss.data[0]))
        train_loss.append(loss_to_print)
        print (epoch, loss_to_print)
        return train_loss
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
        if (epoch == epochs):
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
        test_losses.append(test_loss)
        test_accuracy.append(100. * correct / len(test_loader.dataset))
        return test_losses
    for epoch in range(1, epochs + 1):
        TRAIN_LOSS = train(epoch)
        TEST_LOSS  = test(epoch)
    return TRAIN_LOSS,TEST_LOSS

train_plots_sig = [[]]*10
test_plots_sig = [[]]*10
fig = plt.figure()

for i in range(1, 21,2):
    train_plots_sig[int((i-1)/2)],test_plots_sig[int((i-1)/2)] = solve(F.sigmoid, F.sigmoid, F.sigmoid, 0.05 *i)
    plt.plot(train_plots_sig[int((i-1)/2)],label = "lr_"+str(0.05*i))
plt.legend(loc='lower right')
plt.savefig("training_convergence_sig_sig_sig.png")

fig = plt.figure()

for i in range(1, 21, 2):
    plt.plot(test_plots_sig[int((i-1)/2)],label = "lr_"+str(0.05*i))
plt.legend(loc='lower right')
plt.savefig("test_convergence_sig_sig_sig.png")



train_plots_relu = [[]]*10
test_plots_relu  = [[]]*10
fig = plt.figure()

for i in range(1, 21,2):
    train_plots_relu[int((i-1)/2)],test_plots_relu[int((i-1)/2)] = solve(F.relu, F.relu, F.relu, 0.05 *i)
    plt.plot(train_plots_relu[int((i-1)/2)],label = "lr_"+str(0.05*i))
plt.legend(loc='lower right')
plt.savefig("training_convergence_relu_relu_relu.png")

fig = plt.figure()

for i in range(1, 21, 2):
    plt.plot(test_plots_relu[int((i-1)/2)],label = "lr_"+str(0.05*i))
plt.legend(loc='lower right')
plt.savefig("test_convergence_relu_relu_relu.png")