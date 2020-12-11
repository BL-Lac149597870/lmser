'''
Author: QHGG
Date: 2020-12-08 22:21:58
LastEditTime: 2020-12-09 12:39:38
LastEditors: QHGG
Description: 
FilePath: /lmser/main2.py
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST, CIFAR10, SVHN
from torchvision import transforms
from torch.utils.data import DataLoader
import math
from sklearn.manifold import TSNE
import time

def plot(images, figSize, batchSize, figName):
    fig = plt.figure()
    plt.subplots_adjust(wspace=0, hspace=0)
    a, b = figSize
    for idx in np.arange(batchSize):
        ax = fig.add_subplot(a, b, idx+1)
        ax.imshow(np.squeeze(images[idx]).transpose(1,2,0))
        plt.xticks([]), plt.yticks([])
    plt.savefig(figName)
    plt.close()


def reset_parameters(weight, bias):
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in)
    nn.init.uniform_(bias, -bound, bound)
    return weight, bias


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()
        self.encoderConv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU()
        )
        self.encoderFC = nn.Sequential(
            nn.Linear(64*2*2, 10)
        )
        self.decoderFC = nn.Sequential(
            nn.Linear(10, 64*2*2),
            nn.ReLU()
        )
        self.decoderConv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoderConv(x)
        encoded = encoded.view(-1, 32)
        encoded = self.encoderFC(encoded)
        decoded = self.decoderFC(encoded)
        decoded = decoded.view(-1, 8, 2, 2)
        decoded = self.decoderConv(decoded)
        return decoded


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*2*2, 10)
        )

    def forward(self, x):
        features = self.extractor(x)
        features = features.view(-1, 64*2*2)
        out = self.classifier(features)
        return out


class ConvAutoEncoderManual(nn.Module):
    def __init__(self):
        super(ConvAutoEncoderManual, self).__init__()
        self.weight1 = nn.Parameter(torch.Tensor(16, 3, 3, 3))
        self.bias1 = nn.Parameter(torch.Tensor(16))
        self.weight1, self.bias1 = reset_parameters(self.weight1, self.bias1)
        self.weight2 = nn.Parameter(torch.Tensor(32, 16, 3, 3))
        self.bias2 = nn.Parameter(torch.Tensor(32))
        self.weight2, self.bias2 = reset_parameters(self.weight2, self.bias2)
        self.weight3 = nn.Parameter(torch.Tensor(64, 32, 3, 3))
        self.bias3 = nn.Parameter(torch.Tensor(64))
        self.weight3, self.bias3 = reset_parameters(self.weight3, self.bias3)
        self.weight4 = nn.Parameter(torch.Tensor(64, 64, 3, 3))
        self.bias4 = nn.Parameter(torch.Tensor(64))
        self.weight4, self.bias4 = reset_parameters(self.weight4, self.bias4)
        self.weight5 = nn.Parameter(torch.Tensor(10, 64*2*2))
        self.bias5 = nn.Parameter(torch.Tensor(10))
        self.weight5, self.bias5 = reset_parameters(self.weight5, self.bias5)
        self.weight6 = nn.Parameter(torch.Tensor(64*2*2, 10))
        self.bias6 = nn.Parameter(torch.Tensor(64*2*2))
        self.weight6, self.bias6 = reset_parameters(self.weight6, self.bias6)
        self.weight7 = nn.Parameter(torch.Tensor(64, 64, 3, 3))
        self.bias7 = nn.Parameter(torch.Tensor(64))
        self.weight7, self.bias7 = reset_parameters(self.weight7, self.bias7)
        self.weight8 = nn.Parameter(torch.Tensor(64, 32, 3, 3))
        self.bias8 = nn.Parameter(torch.Tensor(32))
        self.weight8, self.bias8 = reset_parameters(self.weight8, self.bias8)
        self.weight9 = nn.Parameter(torch.Tensor(32, 16, 3, 3))
        self.bias9 = nn.Parameter(torch.Tensor(16))
        self.weight9, self.bias9 = reset_parameters(self.weight9, self.bias9)
        self.weight10 = nn.Parameter(torch.Tensor(16, 3, 3, 3))
        self.bias10 = nn.Parameter(torch.Tensor(3))
        self.weight10, self.bias10 = reset_parameters(self.weight10, self.bias10)

    def forward(self, x):

        h1 = nn.functional.relu(nn.functional.conv2d(input=x, weight=self.weight1, bias=self.bias1,
                                                     stride=3, padding=2))
        h2 = nn.functional.relu(nn.functional.conv2d(input=h1, weight=self.weight2, bias=self.bias2,
                                                     stride=3, padding=0))
        h3 = nn.functional.relu(nn.functional.conv2d(input=h2, weight=self.weight3, bias=self.bias3,
                                                     stride=3, padding=1))
        h4 = nn.functional.relu(nn.functional.conv2d(input=h3, weight=self.weight4, bias=self.bias4,
                                                     stride=1, padding=1))
        h4 = h4.view(-1, 64*2*2)
        h5 = nn.functional.linear(input=h4, weight=self.weight5, bias=self.bias5)
        h6 = nn.functional.relu(nn.functional.linear(input=h5, weight=self.weight5.t(), bias=self.bias6) + h4)
        h6 = h6.view(-1, 64, 2, 2)
        h7 = nn.functional.relu(nn.functional.conv_transpose2d(input=h6, weight=self.weight4, bias=self.bias7,
                                                               stride=1, padding=1) + h3)
        h8 = nn.functional.relu(nn.functional.conv_transpose2d(input=h7, weight=self.weight3, bias=self.bias8,
                                                               stride=3, padding=1) + h2)
        h9 = nn.functional.relu(nn.functional.conv_transpose2d(input=h8, weight=self.weight2, bias=self.bias9,
                                                               stride=3, padding=0) + h1)
        h10 = nn.functional.relu(nn.functional.conv_transpose2d(input=h9, weight=self.weight1, bias=self.bias10,
                                                                stride=3, padding=2))
        return h5, h10

def timeLable():
    return  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

if __name__ == '__main__':
    dataTransform = transforms.Compose([transforms.ToTensor()])
    trainDataset = CIFAR10(root='./data', train=True, transform=dataTransform, download=False)
    testDataset = CIFAR10(root='./data', train=False, transform=dataTransform, download=False)
    batchSize = 64
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False)

    dataIter = iter(testLoader)
    displayImages, displayLabels = dataIter.next()
    displayImages = displayImages.numpy()
    print(displayImages.shape)
    plot(displayImages, (4, 4), 16, './output2/original.png')
    displayImages = torch.tensor(displayImages)
    displayImages = Variable(displayImages.cuda())

    LR = 1e-3
    GAMMA = 0.9
    EPOCH = 30
    RATIO = 0.5
    model = ConvAutoEncoderManual().cuda()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=GAMMA)
    lossFunc1 = nn.MSELoss()
    lossFunc2 = nn.CrossEntropyLoss()
    trainLosses = []
    testLosses = []
    trainReconLosses = []
    testReconLosses = []
    trainPredLosses = []
    testPredLosses = []
    testAcc = []
    for epoch in range(EPOCH):
        epochLoss = 0.0
        epochReconLoss = 0.0
        epochPredLoss = 0.0
        model.train(mode=True)
        for step, (images, labels) in enumerate(trainLoader):
            inputs = Variable(images.cuda())
            outputs = Variable(images.cuda())
            labels = Variable(labels.cuda())
            out, decoded = model(inputs)
            reconLoss = lossFunc1(decoded, outputs)
            predLoss = lossFunc2(out, labels)
            loss = reconLoss + RATIO * predLoss
            epochLoss += loss.cpu().data.numpy()
            epochReconLoss += reconLoss.cpu().data.numpy()
            epochPredLoss += predLoss.cpu().data.numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epochLoss /= (step + 1)
        epochReconLoss /= (step + 1)
        epochPredLoss /= (step + 1)
        trainLosses.append(epochLoss)
        trainReconLosses.append(epochReconLoss)
        trainPredLosses.append(epochPredLoss)
        testLoss = 0.0
        epochReconLoss = 0.0
        epochPredLoss = 0.0
        correct = 0.0
        model.eval()
        for step, (images, labels) in enumerate(testLoader):
            inputs = Variable(images.cuda())
            outputs = Variable(images.cuda())
            labels = Variable(labels.cuda())
            out, decoded = model(inputs)
            reconLoss = lossFunc1(decoded, outputs)
            predLoss = lossFunc2(out, labels)
            loss = reconLoss + RATIO * predLoss
            testLoss += loss.cpu().data.numpy()
            epochReconLoss += reconLoss.cpu().data.numpy()
            epochPredLoss += predLoss.cpu().data.numpy()
            _, pred = out.max(1)
            correct += (pred == labels).sum().cpu().data.numpy()
        testLoss /= (step + 1)
        epochReconLoss /= (step + 1)
        epochPredLoss /= (step + 1)
        correct /= 10000
        testLosses.append(testLoss)
        testReconLosses.append(epochReconLoss)
        testPredLosses.append(epochPredLoss)
        testAcc.append(correct)
        print('Train Epoch {}, Train Loss {:.3f}, Test Loss {:.3f}'.format(epoch + 1, epochLoss, testLoss))
        out, decoded = model(displayImages)
        decoded = decoded.cpu().data.numpy()
        decoded = decoded.reshape(-1, 3, 32, 32)
        plot(decoded, (4, 4), 16, './output2/Epoch_' + str(epoch + 1) + '.png')
        scheduler.step()


    x = [i for i in range(1, EPOCH + 1)]
    line1, = plt.plot(x, trainLosses, 'r')
    line2, = plt.plot(x, testLosses, 'g')
    plt.legend(handles=[line1, line2], labels=['Train Loss', 'Test Loss'])
    plt.title('Train Loss and Test Loss')
    plt.savefig('./CIFAR10/'+ timeLable() +'loss.png')
    plt.close()
    line3, = plt.plot(x, trainReconLosses, 'r')
    line4, = plt.plot(x, testReconLosses, 'g')
    plt.legend(handles=[line3, line4], labels=['Train Recon Loss', 'Test Recon Loss'])
    plt.title('Train Recon Loss and Test Recon Loss')
    plt.savefig('./CIFAR10/'+ timeLable() +'reconLoss.png')
    plt.close()
    line5, = plt.plot(x, trainPredLosses, 'r')
    line6, = plt.plot(x, testPredLosses, 'g')
    plt.legend(handles=[line5, line6], labels=['Train Predict Loss', 'Test Predict Loss'],)
    plt.title('Train Predict Loss and Test Predict Loss')
    plt.savefig('./CIFAR10/'+ timeLable() +'perdLoss.png')
    plt.close()
    plt.plot(x, testAcc)
    plt.title('Test Accuracy')
    plt.savefig('./CIFAR10/'+ timeLable() +'testAcc.png')
    plt.close()
    displayImages = displayImages.cpu().data.numpy()
    displayImages = displayImages.reshape(-1, 3, 32, 32)
    for i in range(displayImages.shape[0]):
        for m in range(16, 32):
            for n in range(32):
                displayImages[i, 0, m, n] = 0
                displayImages[i, 1, m, n] = 0
                displayImages[i, 2, m, n] = 0
    plot(displayImages, (4, 4), 16, './output2/masked.png')
    
    displayImages = torch.tensor(displayImages)
    displayImages = Variable(displayImages.cuda())
    out, decoded = model(displayImages)
    decoded = decoded.cpu().data.numpy()
    decoded = decoded.reshape(-1, 3, 32, 32)
    plot(decoded, (4, 4), 16, './CIFAR10/'+ timeLable() +'recovered.png')
    encoded = []
    encodedLabel = []
    for step, (images, labels) in enumerate(trainLoader):
        inputs = Variable(images.cuda())
        outputs = Variable(images.cuda())
        out, decoded = model(inputs)
        out = out.cpu().data.numpy()
        for i in range(len(labels)):
            encoded.append(out[i])
            encodedLabel.append(labels[i])
    visualizeData = []
    visualizeLabel = []
    
    for i in range(10):
        digitIndex = np.array([j for j in range(50000) if encodedLabel[j] == i])
        np.random.shuffle(digitIndex)
        for k in range(100):
            visualizeData.append(encoded[digitIndex[k]])
            visualizeLabel.append(i)
    visualizeData = np.array(visualizeData)
    visualizeLabel = np.array(visualizeLabel)
    tsne = TSNE(n_components=2)
    visualizeData = tsne.fit_transform(visualizeData)
    colorBoard = ['r', 'g', 'b', 'k', 'm', 'y', 'c', 'peru', 'purple', 'navy']
    for i in range(10):
        paintData = visualizeData[visualizeLabel == i]
        plt.scatter(paintData[:, 0], paintData[:, 1], s=10, c=colorBoard[i])
    plt.savefig('./CIFAR10/'+ timeLable() +'visualize.png')
    plt.close()
