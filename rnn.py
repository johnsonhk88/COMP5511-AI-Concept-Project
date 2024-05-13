import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision

from torch.autograd import Variable

from sklearn.metrics import accuracy_score, confusion_matrix,  classification_report

import matplotlib.pyplot as plt
import datetime

import memory_profiler 

import psutil

'''
STEP 1: LOADING DATASET
'''
train_dataset = dsets.MNIST(root='./data', 
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data', 
                           train=False, 
                           transform=transforms.ToTensor())

'''
STEP 2: MAKING DATASET ITERABLE
'''

batch_size = 100
n_iters = 600#3000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

EnableGPU = True

print("PyTorch Version: ", torch.__version__)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# CPU to GPU
if torch.cuda.is_available() and EnableGPU:
  #  tensor_cpu.cuda()
    print("GPU is available")
    device = torch.device("cuda:0") # Uncomment this to run on GPU
    print("GPU Name: ", torch.cuda.get_device_name())
else: 
    print('No GPU')


# plot one example
print(train_dataset.train_data.size())                 # (60000, 28, 28)
print(train_dataset.train_labels.size())               # (60000)
plt.imshow(train_dataset.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_dataset.train_labels[0])
plt.show()

# pick 2000 samples to speed up testing
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
if torch.cuda.is_available() and EnableGPU:
    test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:10000].cuda()/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
    test_y = test_data.test_labels[:10000].cuda()
else:
    test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:10000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
    test_y = test_data.test_labels[:10000]

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
layer_dim = 2  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
output_dim = 10

StartTrainTime = datetime.datetime.now()
model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)

#######################
#  USE GPU FOR MODEL  #
#######################

if torch.cuda.is_available() and EnableGPU:
    model.cuda()
    
'''
STEP 5: INSTANTIATE LOSS CLASS
'''
criterion = nn.CrossEntropyLoss()

'''
STEP 6: INSTANTIATE OPTIMIZER CLASS
'''
learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

'''
STEP 7: TRAIN THE MODEL
'''

# Number of steps to unroll
seq_dim = 28  


resultStep =[]
resultAcc = []
resultEpoch = []
resultLoss = []
resultCPU = []
resultMem = []

iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Load images as Variable
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        if torch.cuda.is_available() and EnableGPU:
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
                if torch.cuda.is_available() and EnableGPU:
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
                if torch.cuda.is_available() and EnableGPU:
                    correct += (predicted.cpu() == labels.cpu()).sum()
                else:
                    correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / total
            resultAcc.append(accuracy)
            resultStep.append(iter)
            resultEpoch.append(epoch)
            resultLoss.append(loss.data)
            resultCPU.append(psutil.cpu_percent(interval=None, percpu=False))
            resultMem.append(mem_usage[0])
            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {} %'.format(iter, loss.data, accuracy))
          #  print('CPU Usage: ', psutil.cpu_percent(), "%")
          #  print('Memory Usage : {:.2f} MB'.format(mem_usage[0]))
          #  print(psutil.virtual_memory())  # physical memory usage
          #  print('memory % used:', psutil.virtual_memory()[2])


StopTrainTime = datetime.datetime.now()- StartTrainTime
StartTestTime = datetime.datetime.now()
# print 10 predictions from test data
test_output = model(test_x[:10000].view(-1, 28, 28))
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
    print(test_y[:10000], 'real number')
    print("Accuracy: {:.3f}".format(accuracy_score(test_y[:10000], pred_y)))
    print("Confusion matrix:\n{}".format(confusion_matrix(test_y[:10000], pred_y)))
    print("\n\r Classification Report: \n", classification_report(test_y[:10000], pred_y,
                                target_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]))            
            
            
StopTestTime = datetime.datetime.now()- StartTestTime


print('Learning Rate: %.3f' %learning_rate)
print('List of step: ', resultStep,  sep = ', ' )
print('List of Accuracy : ' ,resultAcc, sep = ', ')
print('List of Loss : ', resultLoss, sep=', ')
print('List of CPU Usage : ', resultCPU, sep=', ')
print('List of Memory Usage : ', resultMem, sep=', ')

plt.plot(resultStep, resultAcc)
plt.xlabel('Step (Training)')
plt.ylabel('Accurate (%)')
plt.title('Accuarcy Vs Number of Step')
plt.show()

plt.plot(resultStep, resultLoss)
plt.xlabel('Step (Training)')
plt.ylabel('Loss (%)')
plt.title('Loss Vs Number of Step')
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
