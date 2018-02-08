# Code influenced by: https://github.com/pytorch/examples/blob/master/mnist/main.py

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
epochs = 5
lr = 0.01
momentum = 0.5
no_cuda = False
log_interval = 10

cuda = not no_cuda and torch.cuda.is_available()


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
    def __init__(self, f1, f2, f3, f4):
        super(MLPNetModified, self).__init__()
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.f4 = f4
        self.fc1 = nn.Linear(3*32*32, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 256)
        self.fc4 = nn.Linear(256, 10)
        # self.ceriation = nn.CrossEntropyLoss()
    def forward(self, x):
        x = x.view(-1, 3*32*32)
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
        x = self.fc4(x)
        half = int(len(x[0])/2)
        first_part = x[:, 0:half]
        second_part = x[:, half:]
        first_part = self.f4(first_part)
        second_part = self.f4(second_part)
        x = torch.cat((first_part, second_part), 1)
        # loss = self.ceriation(x, target)
        return F.log_softmax(x)
    def name(self):
        return 'mlpnet'

plots_test_loss = []
plots_train_loss = []
plots_test_accuracy = []

def solve(f1, f2, f3, f4):
    print (str(f1).split()[1], str(f2).split()[1], str(f3).split()[1],str(f4).split()[1])
    model = MLPNetModified(f1, f2, f3, f4)
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
for a in [F.relu, F.sigmoid]:
    for b in [F.relu, F.sigmoid]:
        for c in [F.relu, F.sigmoid]:
            for d in [F.relu, F.sigmoid]:
                solve(a, b, c, d)

# relu relu relu relu
# 1 22432.658615648746
# 2 18264.967773601413
# 3 16744.154054682702
# 4 15527.061487361789
# 5 14488.734002931044

# Test set: Average loss: 1.3557, Accuracy: 5235/10000 (52%)

# relu relu relu sigmoid
# 1 25544.434243559837
# 2 24124.721824884415
# 3 23674.68254840374
# 4 23333.46122598648
# 5 23099.81443655491

# Test set: Average loss: 1.8608, Accuracy: 4743/10000 (47%)

# relu relu sigmoid relu
# 1 28782.55095410347
# 2 28782.31406211853
# 3 28782.31406211853
# 4 28782.31406211853
# 5 28782.31406211853

# Test set: Average loss: 2.3026, Accuracy: 1000/10000 (10%)

# relu relu sigmoid sigmoid
# 1 26773.769926190376
# 2 25305.692128300667
# 3 24690.485558509827
# 4 24060.136650323868
# 5 23701.197252631187

# Test set: Average loss: 1.9032, Accuracy: 4305/10000 (43%)

# relu sigmoid relu relu
# 1 28782.77256989479
# 2 28782.31406211853
# 3 28782.31406211853
# 4 28782.31406211853
# 5 28782.31406211853

# Test set: Average loss: 2.3026, Accuracy: 1000/10000 (10%)

# relu sigmoid relu sigmoid
# 1 26723.455031991005
# 2 25307.713614821434
# 3 24731.8367511034
# 4 24062.095233678818
# 5 23732.22328877449

# Test set: Average loss: 1.8996, Accuracy: 4088/10000 (41%)

# relu sigmoid sigmoid relu
# 1 28782.621402978897
# 2 28782.31406211853
# 3 28782.31406211853
# 4 28782.31406211853
# 5 28782.31406211853

# Test set: Average loss: 2.3026, Accuracy: 1000/10000 (10%)

# relu sigmoid sigmoid sigmoid
# 1 28207.622113347054
# 2 26226.851788640022
# 3 25730.963604211807
# 4 25343.90039038658
# 5 25166.691113829613

# Test set: Average loss: 2.0094, Accuracy: 2513/10000 (25%)

# sigmoid relu relu relu
# 1 28782.511703014374
# 2 28782.31406211853
# 3 28782.31406211853
# 4 28782.31406211853
# 5 28782.31406211853

# Test set: Average loss: 2.3026, Accuracy: 1000/10000 (10%)

# sigmoid relu relu sigmoid
# 1 26626.31323158741
# 2 25618.73391997814
# 3 25290.54021537304
# 4 24901.11425960064
# 5 24610.806240558624

# Test set: Average loss: 1.9627, Accuracy: 3673/10000 (37%)

# sigmoid relu sigmoid relu
# 1 28782.93171977997
# 2 28782.31406211853
# 3 28782.31406211853
# 4 28782.31406211853
# 5 28782.31406211853

# Test set: Average loss: 2.3026, Accuracy: 1000/10000 (10%)

# sigmoid relu sigmoid sigmoid
# 1 27649.928960084915
# 2 26397.039303541183
# 3 26148.455160737038
# 4 25730.365039110184
# 5 25568.36056649685

# Test set: Average loss: 2.0404, Accuracy: 2503/10000 (25%)

# sigmoid sigmoid relu relu
# 1 28782.389269590378
# 2 28782.31406211853
# 3 28782.31406211853
# 4 28782.31406211853
# 5 28782.31406211853

# Test set: Average loss: 2.3026, Accuracy: 1000/10000 (10%)

# sigmoid sigmoid relu sigmoid
# 1 27832.866680383682
# 2 26428.546827316284
# 3 26310.15097117424
# 4 26046.111119270325
# 5 25700.921842575073

# Test set: Average loss: 2.0575, Accuracy: 2133/10000 (21%)

# sigmoid sigmoid sigmoid relu
# 1 28782.479066848755
# 2 28782.31406211853
# 3 28782.31406211853
# 4 28782.31406211853
# 5 28782.31406211853

# Test set: Average loss: 2.3026, Accuracy: 1000/10000 (10%)

# sigmoid sigmoid sigmoid sigmoid
# 1 28820.485750198364
# 2 27939.163905620575
# 3 26527.255042552948
# 4 26377.238058447838
# 5 26319.965564250946

# Test set: Average loss: 2.1049, Accuracy: 1715/10000 (17%)

