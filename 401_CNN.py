"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
torch: 0.4
torchvision
matplotlib
"""
# library
# standard library
import os

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable

from sklearn.metrics import accuracy_score, confusion_matrix,  classification_report

import matplotlib.pyplot as plt
import datetime

import memory_profiler  

import psutil

#import plotly.graph_objects as go

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 100 #50
LR = 0.01              # learning rate
DOWNLOAD_MNIST = False

EnableGPU = False

# CPU to GPU
if torch.cuda.is_available() and EnableGPU:
  #  tensor_cpu.cuda()
    torch.cuda.empty_cache()
    print("GPU is available")
    device = torch.device("cuda:0") # Uncomment this to run on GPU
    print("GPU Name: ", torch.cuda.get_device_name())
    
else: 
    print('No GPU')

print("PyTorch Version: ", torch.__version__)

# Mnist digits dataset
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)

# plot one example
print(train_data.train_data.size())                 # (60000, 28, 28)
print(train_data.train_labels.size())               # (60000)
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# pick 2000 samples to speed up testing
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
if torch.cuda.is_available() and EnableGPU:
    test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:10000].cuda()/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
    test_y = test_data.test_labels[:10000].cuda()
else:
    test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:10000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
    test_y = test_data.test_labels[:10000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #Connvolution 1
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        #Connvolution 2
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    #Make connection
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        #Resize
        #original size: (100, 32, 7, 7)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization


StartTrainTime = datetime.datetime.now()
cnn = CNN()
print(cnn)  # net architecture
if torch.cuda.is_available() and EnableGPU:
    cnn.cuda()
    
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# following function (plot_with_labels) is for visualization, can be ignored if not interested
from matplotlib import cm
try: from sklearn.manifold import TSNE; HAS_SK = False
except: HAS_SK = False; print('Please install sklearn for layer visualization')
def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)



resultStep =[]
resultAcc = []
resultEpoch = []
resultLoss = []
resultCPU = []
resultMem = []


plt.ion()
# training and testing epoch
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader

          #######################
        #  USE GPU FOR MODEL  #
        #######################
        if torch.cuda.is_available() and EnableGPU:
            b_x = Variable(b_x.cuda())
            b_y = Variable(b_y.cuda())
        else:
            b_x = Variable(b_x)
            b_y = Variable(b_y)

        output = cnn(b_x)[0]               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step % 50 == 0:
            test_output, last_layer = cnn(test_x)
            if torch.cuda.is_available() and EnableGPU:
                pred_y = torch.max(test_output, 1)[1].cuda().data
                accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
            else:
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            
            mem_usage = memory_profiler.memory_usage(proc=-1, interval=.2, timeout=None)
            if torch.cuda.is_available() and EnableGPU:
                print('Step: ', step, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.3f %%' %(accuracy*100))
                resultAcc.append(accuracy.numpy()*100)
                resultLoss.append(loss.data)
            else:
                print('Step: ', step, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.3f %%' % (accuracy*100))
                resultAcc.append(accuracy*100)
                resultLoss.append(loss.data.numpy())
            
            resultStep.append(step)
            resultEpoch.append(epoch)
            resultCPU.append(psutil.cpu_percent(interval=None, percpu=False))
            resultMem.append(mem_usage[0])

            #('CPU Usage: ', psutil.cpu_percent(), "%")
            #print('Memory Usage : {:.2f} MB'.format(mem_usage[0]))
            #print(psutil.virtual_memory())  # physical memory usage
            #print('memory % used:', psutil.virtual_memory()[2])
            
            if HAS_SK:
                # Visualization of trained flatten layer (T-SNE)
                tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                plot_only = 500
                if torch.cuda.is_available() and EnableGPU:
                    low_dim_embs = tsne.fit_transform(last_layer.data.cpu().numpy()[:plot_only, :])
                    labels = test_y.data.cpu().numpy()[:plot_only]
                    plot_with_labels(low_dim_embs, labels)
                else:
                    low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                    labels = test_y.numpy()[:plot_only]
                    plot_with_labels(low_dim_embs, labels)
                    
                    
plt.ioff()

StopTrainTime = datetime.datetime.now()- StartTrainTime
StartTestTime = datetime.datetime.now()
# print 1000 predictions from test data
test_output, _ = cnn(test_x[:10000])
if torch.cuda.is_available() and EnableGPU:
    pred_y = torch.max(test_output, 1)[1].cuda().data
    print(pred_y, 'prediction number')
    print(test_y[:10000], 'real number')
    print("Accuracy: {:.3f}".format(accuracy_score(test_y[:10000].cpu().numpy(), pred_y.cpu().numpy())))
    print("Confusion matrix:\n{}".format(confusion_matrix(test_y[:10000].cpu().numpy(), pred_y.cpu().numpy())))
    print("\n\r Classification Report: \n", classification_report(test_y[:10000].cpu().numpy(), pred_y.cpu().numpy(),
                                target_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]))
else:
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    print(pred_y, 'prediction number')
    print(test_y[:10000].numpy(), 'real number')
    print("Accuracy: {:.3f}".format(accuracy_score(test_y[:10000], pred_y)))
    print("Confusion matrix:\n{}".format(confusion_matrix(test_y[:10000], pred_y)))
    print("\n\r Classification Report: \n", classification_report(test_y[:10000], pred_y,
                                target_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]))

StopTestTime = datetime.datetime.now()- StartTestTime
 
    
print('Learning Rate: %.3f' %LR)
print('List of step: ', resultStep,  sep = ', ' )
print('List of Accuracy : ' ,resultAcc, sep = ', ')
print('List of Loss : ', resultLoss, sep=', ')
print('List of CPU Usage : ', resultCPU, sep=', ')
print('List of Memory Usage : ', resultMem, sep=', ')

plt.plot(resultStep, resultAcc)
plt.xlabel('Step (Training)')
plt.ylabel('Accurate (%)')
plt.title('Accuarcy')
plt.show()

plt.plot(resultStep, resultLoss)
plt.xlabel('Step (Training)')
plt.ylabel('Loss (%)')
plt.title('Loss Percentage')
plt.show()

plt.plot(resultStep, resultCPU)
plt.xlabel('Step (Training)')
plt.ylabel('CPU (%)')
plt.title('CPU Usage')
plt.show()

plt.plot(resultStep, resultMem)
plt.xlabel('Step (Training)')
plt.ylabel('Memory (MB)')
plt.title('Memory Usage')
plt.show()

print("\n\rTrain Time : ", StopTrainTime.total_seconds(), "s")
print("Test Time :", StopTestTime.total_seconds() , "s")

if torch.cuda.is_available() and EnableGPU:
    torch.cuda.empty_cache() 
