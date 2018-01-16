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

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class MLPNetModified(nn.Module):

   def __init__(self, f1, f2, f3):
       super(MLPNetModified, self).__init__()
       self.f1 = f1
       self.f2 = f2
       self.f3 = f3
       # 1 input image channel, 6 output channels, 5x5 square convolution
       # kernel
       self.conv1 = nn.Conv2d(3, 6, 5)
       self.conv2 = nn.Conv2d(6, 16, 5)
       # an affine operation: y = Wx + b
       self.fc1 = nn.Linear(16 * 5 * 5, 120)
       self.fc2 = nn.Linear(120, 84)
       self.fc3 = nn.Linear(84, 10)

   def forward(self, x):
       # Max pooling over a (2, 2) window
       x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
       # If the size is a square you can only specify a single number
       x = F.max_pool2d(self.f1(self.conv2(x)), 2)
       x = x.view(-1, self.num_flat_features(x))
       x = self.f1(self.fc1(x))
       x = self.f2(self.fc2(x))
       x = self.fc3(x)
       return F.log_softmax(x)
   def num_flat_features(self, x):
       size = x.size()[1:]  # all dimensions except the batch dimension
       num_features = 1
       for s in size:
           num_features *= s
       return num_features

plots_test_loss = []
plots_train_loss = []
plots_test_accuracy = []

def solve(f1, f2, f3):
    print (str(f1).split()[1], str(f2).split()[1], str(f3).split()[1])
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
            loss_to_print += loss.data[0]
            if batch_idx % log_interval == 0:
                train_loss.append(loss.data[0])
                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, batch_idx * len(data), len(train_loader.dataset),
                #     100. * batch_idx / len(train_loader), loss.data[0]))
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
        if (epoch == epochs):
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
        test_losses.append(test_loss)
        test_accuracy.append(100. * correct / len(test_loader.dataset))
    for epoch in range(1, epochs + 1):
        train(epoch)
        test(epoch)
    # fig = plt.figure()
    # plt.plot(train_loss)
    plots_train_loss.append([str(f1).split()[1]+'_'+str(f2).split()[1]+'_'+str(f3).split()[1]+'_'+'test_loss' + '.png', train_loss])
    # fig.savefig(str(f1).split()[1]+'_'+str(f2).split()[1]+'_'+str(f3).split()[1]+'_'+'train_loss' + '.png', dpi=fig.dpi)
    # fig = plt.figure()
    # plt.plot(test_losses)
    plots_test_loss.append([str(f1).split()[1]+'_'+str(f2).split()[1]+'_'+str(f3).split()[1]+'_'+'test_loss' + '.png', test_losses])
    # fig.savefig(str(f1).split()[1]+'_'+str(f2).split()[1]+'_'+str(f3).split()[1]+'_'+'test_loss' + '.png', dpi=fig.dpi)
    # fig = plt.figure()
    # plt.plot(test_accuracy)
    plots_test_accuracy.append([str(f1).split()[1]+'_'+str(f2).split()[1]+'_'+str(f3).split()[1]+'_'+'test_loss' + '.png', test_accuracy])
    # fig.savefig(str(f1).split()[1]+'_'+str(f2).split()[1]+'_'+str(f3).split()[1]+'_'+'test_accu' + '.png', dpi=fig.dpi)

# plt.show()
for a in [F.relu, F.tanh, F.sigmoid, F.selu]:
    for b in [F.relu, F.tanh, F.sigmoid, F.selu]:
        for c in [F.relu, F.tanh, F.sigmoid, F.selu]:
            solve(a, b, c)


fig = plt.figure()

for a in plots_test_accuracy:
    if ('relu' in a[0]):
        plt.plot(a[1], label='_'.join(a[0].split('_')[0:3]))

plt.legend(loc='lower right')

test_accuracy_last = []

for a in plots_test_accuracy:
    test_accuracy_last.append(['_'.join(a[0].split('_')[0:3]), a[1][len(a[1]) - 1]])


test_accuracy_last.sort(key=lambda x: x[1])
for a in test_accuracy_last:
    print(a)
