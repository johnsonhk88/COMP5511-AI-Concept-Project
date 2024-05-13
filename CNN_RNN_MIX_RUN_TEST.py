# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 23:14:01 2019

@author: GS63
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 21:44:44 2019

@author: GS63
"""

# library
# standard library
import os

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision
from torch.autograd import Variable

from sklearn.metrics import accuracy_score, confusion_matrix,  classification_report

import matplotlib.pyplot as plt
import datetime

import memory_profiler  

import psutil

#import plotly.graph_objects as go

# torch.manual_seed(1)    # reproducible


CNNCPUresultStep =[]
CNNCPUresultAcc = []
CNNCPUresultEpoch = []
CNNCPUresultLoss = []
CNNCPUresultCPU = []
CNNCPUresultMem = []

CNNGPUresultStep =[]
CNNGPUresultAcc = []
CNNGPUresultEpoch = []
CNNGPUresultLoss = []
CNNGPUresultCPU = []
CNNGPUresultMem = []

RNNCPUresultStep =[]
RNNCPUresultAcc = []
RNNCPUresultEpoch = []
RNNCPUresultLoss = []
RNNCPUresultCPU = []
RNNCPUresultMem = []

RNNGPUresultStep =[]
RNNGPUresultAcc = []
RNNGPUresultEpoch = []
RNNGPUresultLoss = []
RNNGPUresultCPU = []
RNNGPUresultMem = []

# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 100 #50
LR = 0.01              # learning rate
DOWNLOAD_MNIST = False
EnableGPU = True

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
    
'''
STEP 1: LOADING DATASET
'''
train_dataset = dsets.MNIST(root='./data', 
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)

test_dataset = dsets.MNIST(root='./data', 
                           train=False, 
                           transform=transforms.ToTensor())


batch_size = 100
n_iters = 600#3000
num_epochs = n_iters / (len(train_data) / batch_size)
num_epochs = int(num_epochs)

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# plot one example
print(train_data.train_data.size())                 # (60000, 28, 28)
print(train_data.train_labels.size())               # (60000)
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()



# pick 2000 samples to speed up testing
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

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
    
'''
STEP 3: CREATE MODEL CLASS
''' 
    
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # Building your RNN
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='tanh')
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    
    def forward(self, x):
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        if torch.cuda.is_available() and EnableGPU:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
            
        # One time step
        out, hn = self.rnn(x, h0)
        
        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out    
    
'''
STEP 4: INSTANTIATE MODEL CLASS
'''
input_dim = 28
hidden_dim = 100
layer_dim = 3  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
output_dim = 10
learning_rate = 0.01
    

def CNNRunTrain(GPUEnble):
    if torch.cuda.is_available() and GPUEnble:
        test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:10000].cuda()/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
        test_y = test_data.test_labels[:10000].cuda()
    else:
        test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:10000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
        test_y = test_data.test_labels[:10000]
    StartTrainTime = datetime.datetime.now()
    cnn = CNN()
    print(cnn)  # net architecture
    if torch.cuda.is_available() and GPUEnble:
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





    plt.ion()
    # training and testing epoch
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader

              #######################
            #  USE GPU FOR MODEL  #
            #######################
            if torch.cuda.is_available() and GPUEnble:
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
                if torch.cuda.is_available() and GPUEnble:
                    pred_y = torch.max(test_output, 1)[1].cuda().data
                    accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
                else:
                    pred_y = torch.max(test_output, 1)[1].data.numpy()
                    accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))

                mem_usage = memory_profiler.memory_usage(proc=-1, interval=.2, timeout=None)
                if torch.cuda.is_available() and GPUEnble:
                    print('CNN GPU | Step: ', step, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.3f %%' %(accuracy*100))
                    CNNGPUresultAcc.append(accuracy.numpy()*100)
                    CNNGPUresultLoss.append(loss.data)
                    CNNGPUresultStep.append(step)
                    CNNGPUresultEpoch.append(epoch)
                    CNNGPUresultCPU.append(psutil.cpu_percent(interval=None, percpu=False))
                    CNNGPUresultMem.append(mem_usage[0])
                else:
                    print('CNN CPU | Step: ', step, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.3f %%' % (accuracy*100))
                    CNNCPUresultAcc.append(accuracy*100)
                    CNNCPUresultLoss.append(loss.data.numpy())
                    CNNCPUresultStep.append(step)
                    CNNCPUresultEpoch.append(epoch)
                    CNNCPUresultCPU.append(psutil.cpu_percent(interval=None, percpu=False))
                    CNNCPUresultMem.append(mem_usage[0])

                #resultStep.append(step)
                #resultEpoch.append(epoch)
                #resultCPU.append(psutil.cpu_percent(interval=None, percpu=False))
                #resultMem.append(mem_usage[0])

                #('CPU Usage: ', psutil.cpu_percent(), "%")
                #print('Memory Usage : {:.2f} MB'.format(mem_usage[0]))
                #print(psutil.virtual_memory())  # physical memory usage
                #print('memory % used:', psutil.virtual_memory()[2])

                if HAS_SK:
                    # Visualization of trained flatten layer (T-SNE)
                    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                    plot_only = 500
                    if torch.cuda.is_available() and GPUEnble:
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
    if torch.cuda.is_available() and GPUEnble:
        pred_y = torch.max(test_output, 1)[1].cuda().data
        print(pred_y, 'prediction number')
        print(test_y[:10000], 'real number')
        print("CNN GPU | Accuracy: {:.3f}".format(accuracy_score(test_y[:10000].cpu().numpy(), pred_y.cpu().numpy())))
        print("CNN GPU | Confusion matrix:\n{}".format(confusion_matrix(test_y[:10000].cpu().numpy(), pred_y.cpu().numpy())))
        print("\n\rCNN GPU | Classification Report: \n", classification_report(test_y[:10000].cpu().numpy(), pred_y.cpu().numpy(),
                                    target_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]))
    else:
        pred_y = torch.max(test_output, 1)[1].data.numpy()
        print(pred_y, 'prediction number')
        print(test_y[:10000].numpy(), 'real number')
        print("CNN CPU | Accuracy: {:.3f}".format(accuracy_score(test_y[:10000], pred_y)))
        print("CNN CPU | Confusion matrix:\n{}".format(confusion_matrix(test_y[:10000], pred_y)))
        print("\n\rCNN CPU | Classification Report: \n", classification_report(test_y[:10000], pred_y,
                                    target_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]))

    StopTestTime = datetime.datetime.now()- StartTestTime
    return StopTrainTime, StopTestTime
 

def RNNRunTrain(GPUEnble):
    if torch.cuda.is_available() and GPUEnble:
        test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:10000].cuda()/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
        test_y = test_data.test_labels[:10000].cuda()
    else:
        test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:10000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
        test_y = test_data.test_labels[:10000]
    StartTrainTime = datetime.datetime.now()
    model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)

    #######################
    #  USE GPU FOR MODEL  #
    #######################

    if torch.cuda.is_available() and GPUEnble:
        model.cuda()

    '''
    STEP 5: INSTANTIATE LOSS CLASS
    '''
    criterion = nn.CrossEntropyLoss()

    '''
    STEP 6: INSTANTIATE OPTIMIZER CLASS
    '''
    

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)   # optimize all cnn parameters

    '''
    STEP 7: TRAIN THE MODEL
    '''

    # Number of steps to unroll
    seq_dim = 28  




    iter = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Load images as Variable
            #######################
            #  USE GPU FOR MODEL  #
            #######################
            if torch.cuda.is_available() and GPUEnble:
                images = Variable(images.view(-1, seq_dim, input_dim).cuda())
                labels = Variable(labels.cuda())
            else:
                images = Variable(images.view(-1, seq_dim, input_dim))
                labels = Variable(labels)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            # outputs.size() --> 100, 10
            outputs = model(images)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            iter += 1

            if iter % 50 == 0:
                # Calculate Accuracy         
                correct = 0
                total = 0
                # Iterate through test dataset
                for images, labels in test_loader:
                    #######################
                    #  USE GPU FOR MODEL  #
                    #######################
                    if torch.cuda.is_available() and GPUEnble:
                        images = Variable(images.view(-1, seq_dim, input_dim).cuda())
                    else:
                        images = Variable(images.view(-1, seq_dim, input_dim))

                    # Forward pass only to get logits/output
                    outputs = model(images)

                    # Get predictions from the maximum value
                    _, predicted = torch.max(outputs.data, 1)

                    # Total number of labels
                    total += labels.size(0)

                    # Total correct predictions
                    #######################
                    #  USE GPU FOR MODEL  #
                    #######################
                    mem_usage = memory_profiler.memory_usage(proc=-1, interval=.1, timeout=None)
                    if torch.cuda.is_available() and GPUEnble:
                        correct += (predicted.cpu() == labels.cpu()).sum()
                    else:
                        correct += (predicted == labels).sum()

                accuracy = 100 * correct / total
                if torch.cuda.is_available() and GPUEnble:
                    RNNGPUresultAcc.append(accuracy)
                    RNNGPUresultStep.append(iter)
                    RNNGPUresultEpoch.append(epoch)
                    RNNGPUresultLoss.append(loss.data)
                    RNNGPUresultCPU.append(psutil.cpu_percent(interval=None, percpu=False))
                    RNNGPUresultMem.append(mem_usage[0])
                    # Print Loss
                    print('RNN GPU | Iteration: {}. Loss: {}. Accuracy: {} %'.format(iter, loss.data, accuracy))
                else:
                    RNNCPUresultAcc.append(accuracy)
                    RNNCPUresultStep.append(iter)
                    RNNCPUresultEpoch.append(epoch)
                    RNNCPUresultLoss.append(loss.data)
                    RNNCPUresultCPU.append(psutil.cpu_percent(interval=None, percpu=False))
                    RNNCPUresultMem.append(mem_usage[0])
                    # Print Loss
                    print('RNN CPU | Iteration: {}. Loss: {}. Accuracy: {} %'.format(iter, loss.data, accuracy))
                #resultAcc.append(accuracy)
                #resultStep.append(iter)
                #resultEpoch.append(epoch)
                #resultLoss.append(loss.data)
                #resultCPU.append(psutil.cpu_percent(interval=None, percpu=False))
                #resultMem.append(mem_usage[0])
                # Print Loss
                #print('Iteration: {}. Loss: {}. Accuracy: {} %'.format(iter, loss.data, accuracy))
              #  print('CPU Usage: ', psutil.cpu_percent(), "%")
              #  print('Memory Usage : {:.2f} MB'.format(mem_usage[0]))
              #  print(psutil.virtual_memory())  # physical memory usage
              #  print('memory % used:', psutil.virtual_memory()[2])


    StopTrainTime = datetime.datetime.now()- StartTrainTime
    StartTestTime = datetime.datetime.now()
    # print 10 predictions from test data
    test_output = model(test_x[:10000].view(-1, 28, 28))
    if torch.cuda.is_available() and GPUEnble:
        pred_y = torch.max(test_output, 1)[1].cuda().data
        print(pred_y, 'prediction number')
        print(test_y[:10000], 'real number')
        print("RNN GPU | Accuracy: {:.3f}".format(accuracy_score(test_y[:10000].cpu().numpy(), pred_y.cpu().numpy())))
        print("RNN GPU | Confusion matrix:\n{}".format(confusion_matrix(test_y[:10000].cpu().numpy(), pred_y.cpu().numpy())))
        print("\n\rRNN GPU | Classification Report: \n", classification_report(test_y[:10000].cpu().numpy(), pred_y.cpu().numpy(),
                                    target_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]))
    else:
        pred_y = torch.max(test_output, 1)[1].data.numpy()
        print(pred_y, 'prediction number')
        print(test_y[:10000], 'real number')
        print("RNN CPU |Accuracy: {:.3f}".format(accuracy_score(test_y[:10000], pred_y)))
        print("RNN CPU |Confusion matrix:\n{}".format(confusion_matrix(test_y[:10000], pred_y)))
        print("\n\rRNN CPU | Classification Report: \n", classification_report(test_y[:10000], pred_y,
                                    target_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]))            


    StopTestTime = datetime.datetime.now()- StartTestTime
    return StopTrainTime , StopTestTime



CNNCPUTrainTime,  CNNCPUTestTime =  CNNRunTrain(False)
CNNGPUTrainTime,  CNNGPUTestTime =  CNNRunTrain(True)
torch.cuda.empty_cache() 
EnableGPU = False
RNNCPUTrainTime , RNNCPUTestTime = RNNRunTrain(False)
EnableGPU = True
RNNGPUTrainTime , RNNGPUTestTime = RNNRunTrain(True)
torch.cuda.empty_cache() 


print("\n\rCNN Result (CPU):")
print('Learning Rate: %.3f' %LR)
print('CNN (CPU) List of step: ', CNNCPUresultStep,  sep = ', ' )
print('CNN (CPU) List of Accuracy : ' ,CNNCPUresultAcc, sep = ', ')
print('CNN (CPU) List of Loss : ', CNNCPUresultLoss, sep=', ')
print('CNN (CPU) List of CPU Usage : ', CNNCPUresultCPU, sep=', ')
print('CNN (CPU) List of Memory Usage : ', CNNCPUresultMem, sep=', ')

print("\n\rCNN Result (GPU):")
print('Learning Rate: %.3f' %LR)
print('CNN (GPU) List of step: ', CNNGPUresultStep,  sep = ', ' )
print('CNN (GPU) List of Accuracy : ' ,CNNGPUresultAcc, sep = ', ')
print('CNN (GPU) List of Loss : ', CNNGPUresultLoss, sep=', ')
print('CNN (GPU) List of CPU Usage : ', CNNGPUresultCPU, sep=', ')
print('CNN (GPU) List of Memory Usage : ', CNNGPUresultMem, sep=', ')

print("\n\rRNN Result (CPU):")
print('Learning Rate: %.3f' %learning_rate)
print('RNN (CPU) List of step: ', RNNCPUresultStep,  sep = ', ' )
print('RNN (CPU) List of Accuracy : ' ,RNNCPUresultAcc, sep = ', ')
print('RNN (CPU) List of Loss : ', RNNCPUresultLoss, sep=', ')
print('RNN (CPU) List of CPU Usage : ', RNNCPUresultCPU, sep=', ')
print('RNN (CPU) List of Memory Usage : ', RNNCPUresultMem, sep=', ')

print("\n\rRNN Result (GPU):")
print('Learning Rate: %.3f' %learning_rate)
print('RNN (GPU) List of step: ', RNNGPUresultStep,  sep = ', ' )
print('RNN (GPU) List of Accuracy : ' ,RNNGPUresultAcc, sep = ', ')
print('RNN (GPU) List of Loss : ', RNNGPUresultLoss, sep=', ')
print('RNN (GPU) List of CPU Usage : ', RNNGPUresultCPU, sep=', ')
print('RNN (GPU) List of Memory Usage : ', RNNGPUresultMem, sep=', ')


#result ploting
plt.plot(CNNCPUresultStep, CNNCPUresultAcc , label="CNN CPU")
plt.plot(CNNGPUresultStep, CNNGPUresultAcc , label="CNN GPU")
plt.plot(RNNCPUresultStep, RNNCPUresultAcc , label="RNN CPU")
plt.plot(RNNGPUresultStep, RNNGPUresultAcc , label="RNN GPU")
plt.xlabel('Step (Training)')
plt.ylabel('Accurate (%)')
plt.title('Accuarcy')
plt.legend(loc='best')
plt.show()

plt.plot(CNNCPUresultStep, CNNCPUresultLoss, label="CNN CPU")
plt.plot(CNNGPUresultStep, CNNGPUresultLoss , label="CNN GPU")
plt.plot(RNNCPUresultStep, RNNCPUresultLoss , label="RNN CPU")
plt.plot(RNNGPUresultStep, RNNGPUresultLoss , label="RNN GPU")
plt.xlabel('Step (Training)')
plt.ylabel('Loss (%)')
plt.title('Loss Percentage')
plt.legend(loc='best')
plt.show()

plt.plot(CNNCPUresultStep, CNNCPUresultCPU, label="CNN CPU")
plt.plot(CNNGPUresultStep, CNNGPUresultCPU , label="CNN GPU")
plt.plot(RNNCPUresultStep, RNNCPUresultCPU , label="RNN CPU")
plt.plot(RNNGPUresultStep, RNNGPUresultCPU , label="RNN GPU")
plt.xlabel('Step (Training)')
plt.ylabel('CPU (%)')
plt.title('CPU Usage')
plt.legend(loc='best')
plt.show()


plt.plot(CNNCPUresultStep, CNNCPUresultMem, label="CNN CPU")
plt.plot(CNNGPUresultStep, CNNGPUresultMem , label="CNN GPU")
plt.plot(RNNCPUresultStep, RNNCPUresultMem , label="RNN CPU")
plt.plot(RNNGPUresultStep, RNNGPUresultMem , label="RNN GPU")
plt.xlabel('Step (Training)')
plt.ylabel('Memory (MB)')
plt.title('Memory Usage')
plt.legend(loc='best')
plt.show()

print("\n\rCNN (CPU) Train Time : ", CNNCPUTrainTime.total_seconds(), "s")
print("CNN (CPU) Test Time :", CNNCPUTestTime.total_seconds() , "s")
print("\n\rCNN (GPU) Train Time : ", CNNGPUTrainTime.total_seconds(), "s")
print("CNN (GPU) Test Time :", CNNGPUTestTime.total_seconds() , "s")

print("\n\rRNN (CPU) Train Time : ", RNNCPUTrainTime.total_seconds(), "s")
print("RNN (CPU) Test Time :", RNNCPUTestTime.total_seconds() , "s")
print("\n\rRNN (GPU) Train Time : ", RNNGPUTrainTime.total_seconds(), "s")
print("RNN (GPU) Test Time :", RNNGPUTestTime.total_seconds() , "s")

if torch.cuda.is_available() and EnableGPU:
    torch.cuda.empty_cache() 