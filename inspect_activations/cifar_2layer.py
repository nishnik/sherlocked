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
    def __init__(self, f1, f2):
        super(MLPNetModified, self).__init__()
        self.f1 = f1
        self.f2 = f2
        self.fc1 = nn.Linear(3*32*32, 500)
        self.fc3 = nn.Linear(500, 10)
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
        x = self.fc3(x)
        half = int(len(x[0])/2)
        first_part = x[:, 0:half]
        second_part = x[:, half:]
        first_part = self.f2(first_part)
        second_part = self.f2(second_part)
        x = torch.cat((first_part, second_part), 1)
        # loss = self.ceriation(x, target)
        return F.log_softmax(x)
    def name(self):
        return 'mlpnet'

plots_test_loss = []
plots_train_loss = []
plots_test_accuracy = []

def solve(f1, f2):
    print (str(f1).split()[1], str(f2).split()[1])
    model = MLPNetModified(f1, f2)
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
    # plots_train_loss.append([str(f1).split()[1]+'_'+str(f2).split()[1]+'_'+str(f3).split()[1]+'_'+'test_loss' + '.png', train_loss])
    # fig.savefig(str(f1).split()[1]+'_'+str(f2).split()[1]+'_'+str(f3).split()[1]+'_'+'train_loss' + '.png', dpi=fig.dpi)
    # fig = plt.figure()
    # plt.plot(test_losses)
    # plots_test_loss.append([str(f1).split()[1]+'_'+str(f2).split()[1]+'_'+str(f3).split()[1]+'_'+'test_loss' + '.png', test_losses])
    # fig.savefig(str(f1).split()[1]+'_'+str(f2).split()[1]+'_'+str(f3).split()[1]+'_'+'test_loss' + '.png', dpi=fig.dpi)
    # fig = plt.figure()
    # plt.plot(test_accuracy)
    # plots_test_accuracy.append([str(f1).split()[1]+'_'+str(f2).split()[1]+'_'+str(f3).split()[1]+'_'+'test_loss' + '.png', test_accuracy])
    # fig.savefig(str(f1).split()[1]+'_'+str(f2).split()[1]+'_'+str(f3).split()[1]+'_'+'test_accu' + '.png', dpi=fig.dpi)

# plt.show()
for a in [F.relu, F.tanh, F.sigmoid, F.selu]:
    for b in [F.relu, F.tanh, F.sigmoid, F.selu]:
            solve(a, b)


# relu relu
# 1 22483.001011841
# 2 22114.300386108458
# 3 21788.849059840664
# 4 21489.548542417586
# 5 21125.876043519005

# Test set: Average loss: 1.8367, Accuracy: 4094/10000 (41%)

# relu tanh
# 1 22467.574537217617
# 2 21438.75602531433
# 3 21125.03168386221
# 4 20943.708651661873
# 5 20784.09401959181

# Test set: Average loss: 1.7128, Accuracy: 4503/10000 (45%)

# relu sigmoid
# 1 24897.652915596962
# 2 23988.560150504112
# 3 23621.34576678276
# 4 23356.139699935913
# 5 23145.061411976814

# Test set: Average loss: 1.8770, Accuracy: 4832/10000 (48%)

# relu selu
# 1 22389.9954700619
# 2 21744.196449004114
# 3 21511.717118304223
# 4 21167.789812356466
# 5 21016.8208998912

# Test set: Average loss: 1.8498, Accuracy: 4482/10000 (45%)

# tanh relu
# 1 22742.315711528063
# 2 21278.192411988974
# 3 20728.743396550417
# 4 20282.158412784338
# 5 19948.83623766899

# Test set: Average loss: 1.7168, Accuracy: 4264/10000 (43%)

# tanh tanh
# 1 23610.294448018074
# 2 22431.334889292717
# 3 21846.279523313046
# 4 21414.550306022167
# 5 21034.413868546486

# Test set: Average loss: 1.7311, Accuracy: 4419/10000 (44%)

# tanh sigmoid
# 1 25498.29685294628
# 2 24764.382350325584
# 3 24391.785904288292
# 4 24125.830772042274
# 5 23892.733357310295

# Test set: Average loss: 1.9338, Accuracy: 4261/10000 (43%)

# tanh selu
# 1 22826.763731092215
# 2 21480.958216659725
# 3 20891.62016709894
# 4 20421.850892670453
# 5 20065.322593457997

# Test set: Average loss: 1.7269, Accuracy: 4241/10000 (42%)

# sigmoid relu
# 1 28782.56166577339
# 2 28782.31406211853
# 3 28782.31406211853
# 4 28782.31406211853
# 5 28782.31406211853

# Test set: Average loss: 2.3026, Accuracy: 1000/10000 (10%)

# sigmoid tanh
# 1 24100.898237347603
# 2 22878.596187591553
# 3 22328.387303352356
# 4 21872.452538609505
# 5 21481.960980951786

# Test set: Average loss: 1.7404, Accuracy: 4440/10000 (44%)

# sigmoid sigmoid
# 1 26102.29568183422
# 2 25196.28473997116
# 3 24910.5311126709
# 4 24743.622224092484
# 5 24617.71556019783

# Test set: Average loss: 1.9728, Accuracy: 3938/10000 (39%)

# sigmoid selu
# 1 23463.293425768614
# 2 21316.172017514706
# 3 20224.414303407073
# 4 19278.906116470695
# 5 18440.601242803037

# Test set: Average loss: 1.5564, Accuracy: 4754/10000 (48%)

# selu relu
# 1 23963.959069788456
# 2 23020.72505198419
# 3 22772.651195146143
# 4 22552.09944573231
# 5 22539.026905629784

# Test set: Average loss: 1.8562, Accuracy: 3913/10000 (39%)

# selu tanh
# 1 24227.00868934393
# 2 23422.149784743786
# 3 23163.067794799805
# 4 22860.06668537855
# 5 22771.801949977875

# Test set: Average loss: 1.8282, Accuracy: 3516/10000 (35%)

# selu sigmoid
# 1 25352.930349946022
# 2 24615.741607308388
# 3 24241.96420264244
# 4 23963.812512636185
# 5 23754.276213765144

# Test set: Average loss: 1.9160, Accuracy: 4436/10000 (44%)

# selu selu
# 1 24189.079659238458
# 2 23320.222410470247
# 3 23125.98736745119
# 4 22978.082153767347
# 5 23016.40514079295

# Test set: Average loss: 1.8863, Accuracy: 4041/10000 (40%)

