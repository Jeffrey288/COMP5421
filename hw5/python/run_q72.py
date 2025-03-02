import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

device = torch.device('cpu')

train_data = datasets.MNIST(root='../data', train=True, transform=ToTensor(), download=True)
test_data = datasets.MNIST(root='../data', train=False, transform=ToTensor())
train_x, train_y = train_data.data, train_data.targets
test_x, test_y = test_data.data, test_data.targets
# print(train_x[0].size())
# plt.imshow(train_x[0], cmap='gray')
# plt.show()

batch_size = 80 # note that there are 60000 samples
num_batches = train_x.size(0) // batch_size
max_iter = 2
learning_rate = 5e-2

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

# some reference is taken from LeNet
class CNN(nn.Module): # image size is 28x28
    def __init__(self):
        super().__init__()
        self.first_conv = nn.Conv2d(1, 7, 5) 
        self.second_conv = nn.Conv2d(7, 15, 5)
        self.pool = nn.MaxPool2d(2, 2) 
        self.first_fc = nn.Linear(240, 60)
        self.second_fc = nn.Linear(60, 10)

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
gradient_method = optim.SGD(cnn.parameters(), lr=learning_rate)

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
        loss = loss_function(yhat_probs, y)
        loss.backward()
        gradient_method.step()
        
        correct = torch.eq(yhat, y).sum().item()
        cum_correct += correct
        cum_loss += loss.item() * batch_size # loss is tensor
        train_loss.append(loss.item())
        train_acc.append(correct / batch_size)

        if i % 100 == 0:
            print(f"Epoch {epoch} batch {i} loss: {cum_loss} acc: {cum_correct}")

            # cum_test_correct = 0
            # cum_test_loss = 0
            # with torch.no_grad():
            #     for data in test_loader:
            #         x, y = data
            #         yhat_probs = cnn(x)
            #         _, yhat = torch.max(yhat_probs.data, 1)
            #         loss = loss_function(yhat_probs, y)
            #         cum_test_correct += (yhat == y).sum().item()
            #         cum_test_loss += loss.item()
            # print(f"{cum_test_loss/test_x.size()[0]} {cum_test_correct/test_x.size()[0]}")
    
    train_loss_overall.append(cum_loss / train_x.size()[0])
    train_acc_overall.append(cum_correct / train_x.size()[0])
    print(f"Epoch {epoch} loss: {cum_loss} acc:{cum_correct/train_x.size()[0]}")

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