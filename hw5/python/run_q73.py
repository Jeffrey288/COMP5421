import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import scipy.io
import numpy as np

# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

device = torch.device('cpu')

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

# 1 channel, and tranpose image
train_x = np.reshape(train_x, (train_x.shape[0], 1, 32, 32)).transpose([0, 1, 3, 2])
valid_x = np.reshape(valid_x, (valid_x.shape[0], 1, 32, 32)).transpose([0, 1, 3, 2])
# convert boolean array to categorical index
print(train_y.shape)
train_y = np.sum(train_y * np.array([np.arange(train_y.shape[1], dtype=int)]), axis=1)
valid_y = np.sum(valid_y * np.array([np.arange(valid_y.shape[1], dtype=int)]), axis=1)
print(train_y)

train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
train_y = torch.from_numpy(train_y).type(torch.LongTensor)
test_x = torch.from_numpy(valid_x).type(torch.FloatTensor)
test_y = torch.from_numpy(valid_y).type(torch.LongTensor)
# print("test_y", test_y.dtype)
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class NIST36Dataset(Dataset):

    def __init__(self, x, y):
        assert x.size(0) == y.size(0), 'x and y should have the same no. of samples!'
        self.len = x.size(0)
        self.x = x
        self.y = y
        # print("y", y.dtype)
        # print("self.y", self.y.dtype)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # print("self.y2", self.y.dtype)
        # print("self.y[idx]", self.y[idx].dtype)
        return (self.x[idx], self.y[idx])

train_data = NIST36Dataset(train_x, train_y)
test_data = NIST36Dataset(test_x, test_y)

# for i in range(5):
#     print(train_x[i].shape)
#     plt.imshow(train_x[i][0], cmap='gray')
#     plt.show()

# print(len(train_data))
# input()

batch_size = 80 # 10800 samples
num_batches = train_x.size(0) // batch_size
max_iter = 20
learning_rate = 7e-2
reg_factor = 3e-4

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

class CNN(nn.Module): # image size is 32x32
    def __init__(self):
        super().__init__()
        self.first_conv = nn.Conv2d(1, 7, 5) 
        self.second_conv = nn.Conv2d(7, 15, 5)
        self.pool = nn.MaxPool2d(2, 2) 
        self.first_fc = nn.Linear(375, 180) #
        self.second_fc = nn.Linear(180, 36)

    def forward(self, x):
        x = F.relu(self.first_conv(x))
        x = self.pool(x)
        x = F.relu(self.second_conv(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.first_fc(x))
        x = self.second_fc(x)
        return x

cnn = CNN()

loss_function = nn.CrossEntropyLoss() # better for that
# stochastic (mini-batch) gradient method 
gradient_method = optim.SGD(cnn.parameters(), lr=learning_rate, weight_decay=1e-4) # do regularization here instead

train_loss = []
train_loss_overall = []
train_acc = []
train_acc_overall = []
test_loss = []
test_acc = []
for epoch in range(max_iter + 1):
    
    cum_test_correct = 0
    cum_test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            x, y = data
            yhat_probs = cnn(x)
            _, yhat = torch.max(yhat_probs.data, 1)
            loss = loss_function(yhat_probs, y)
            cum_test_correct += (yhat == y).sum().item()
            cum_test_loss += loss.item()
    test_loss.append(cum_test_loss / test_x.size()[0])
    test_acc.append(cum_test_correct / test_x.size()[0])
    print(f"Epoch {epoch} test: {cum_test_loss/test_x.size()[0]} {cum_test_correct/test_x.size()[0]}")

    if epoch == max_iter: break
    
    cum_loss = 0
    cum_correct = 0
    for i, data in enumerate(train_loader):
        x, y = data

        gradient_method.zero_grad()

        yhat_probs = cnn(x)
        _, yhat = torch.max(yhat_probs, dim=1)
        # print(yhat_probs, y)
        loss = loss_function(yhat_probs, y) # changing type of yhat causes grad to be disabled
        # for param in cnn.parameters():
        #     loss += reg_factor * l1_reg(param, target=torch.zeros_like(param))
        loss.backward()
        gradient_method.step()
        
        correct = torch.eq(yhat, y).sum().item()
        cum_correct += correct
        cum_loss += loss.item() * batch_size # loss is tensor
        train_loss.append(loss.item())
        train_acc.append(correct / batch_size)

        if (i+1) % 18 == 0:
            print(f"Epoch {epoch} batch {i} loss: {cum_loss} acc: {cum_correct}")

    train_loss_overall.append(cum_loss / train_x.size()[0])
    train_acc_overall.append(cum_correct / train_x.size()[0])
    print(f"Epoch {epoch} loss: {cum_loss} acc:{cum_correct/train_x.size()[0]}")

# print(train_loss)

# print(train_loss)
print(train_loss_overall)
print(train_acc_overall)
print(test_loss)
print(test_acc)

fig, ax = plt.subplots(1, 2)
ax[0].plot(np.arange(len(train_loss))/num_batches, train_loss, 'co', markersize=1)
ax[0].plot(np.arange(1, max_iter+1), train_loss_overall, 'b-')
ax[0].plot(np.arange(0, max_iter+1), test_loss, 'r-')
ax[0].legend(['Training Minibatches', 'Training', 'Validation'])
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('average loss')
ax[0].set_title("Average loss")
ax[1].plot(np.arange(len(train_acc))/num_batches, train_acc, 'co', markersize=1)
ax[1].plot(np.arange(1, max_iter+1), train_acc_overall, 'b-')
ax[1].plot(np.arange(0, max_iter+1), test_acc, 'r-')
ax[1].legend(['Training Minibatches', 'Training', 'Validation'])
ax[1].set_xlabel('epoch')
ax[1].set_ylabel('accuracy')
ax[1].set_title("Classification accuracy")
plt.show()
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in test_loader:
#         x, y = data
#         yhat_probs = cnn(y)
#     _, yhat = torch.max(yhat_probs.data, 1)
#     correct += (yhat == y).sum().item()
#     total += y.size(0)

# print(correct, total)