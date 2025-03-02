# import scipy.io
# import numpy as np
# from nn import *
# import torch
# import torchvision
# from torchvision import transforms, datasets

# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# from torch.autograd import Variable
# import skimage
# import skimage.io
# import skimage.transform
# import torch.optim as optim
# import os
# import matplotlib.pyplot as plt
# import matplotlib.patches
# from nn import *
# from q4 import *
# # do not include any more libraries here!
# # no opencv, no sklearn, etc!
# import warnings

# max_iters = 20

# batch_size = 16
# learning_rate = 1e-3
# hidden_size = 64

# trainroot = '../data/oxford-flowers17/train'
# traindirs = [x[1] for x in os.walk(trainroot)][0]
# counter = 0
# print(traindirs)
# # label = []
# # training_data = []
# # for folder in traindirs:
# #     root = trainroot+'/'+folder
# #     for subdirs,dirs,files in os.walk(root):
# #         # print(files)
# #         for file in files:
# #             path = root+ '/' + file
# #             im = skimage.img_as_float(skimage.io.imread(path))
# #             # print(im.shape)
# #             label.append(int(folder))
# #             im = skimage.transform.resize(im,(224,224))
# #             # print(im.shape)
# #             training_data.append(im)
# #
# # training_data = np.array(training_data)
# # np.save('training_data.npy',training_data)
# # np.save('label.npy',label)
# # training_data = np.load('training_data.npy')
# # # print(training_data)
# # label = np.load('label.npy')
# # # print(training_data)
# # examples = training_data.shape[0]
# # print(examples)
# # train_x = torch.tensor(training_data).float()
# # label = torch.tensor(label).long()

# # use image folder to read the data

# data_transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         # transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])


# trainset = datasets.ImageFolder(root='../data/oxford-flowers17/train',
#                           transform=data_transform)
# train_loader = torch.utils.data.DataLoader(trainset,
#                                            batch_size=batch_size,
#                                            shuffle=True)


# testset = datasets.ImageFolder(root='../data/oxford-flowers17/train',
#                           transform=data_transform)
# test_loader = torch.utils.data.DataLoader(testset,
#                                            batch_size=batch_size,
#                                            shuffle=True)

# examples = len(trainset)
# test_examples = len(testset)
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")
# print(use_cuda)

# #self defined
# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = torch.nn.Conv2d(3, 10, 5, 1)
#         self.conv2 = torch.nn.Conv2d(10, 50, 5, 1)
#         self.conv3 = torch.nn.Conv2d(50, 200, 5, 1)
#         self.fc1 = torch.nn.Linear(24*24*200, 512) # was 4*4*50, 500 for
#         self.fc2 = torch.nn.Linear(512, 17) # was 500,10 for mnist

#     def forward(self, x):
#         x = torch.nn.functional.relu(self.conv1(x))
#         x = torch.nn.functional.max_pool2d(x, 2, 2)
#         x = torch.nn.functional.relu(self.conv2(x))
#         x = torch.nn.functional.max_pool2d(x, 2, 2)
#         x = torch.nn.functional.relu(self.conv3(x))
#         x = torch.nn.functional.max_pool2d(x, 2, 2)
#         # print(x.size())

#         x = x.view(-1, 24*24*200)
#         # print(x.size())
#         x = torch.nn.functional.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # self defined net
# model = Net().to(device)


# # output of squeezenet was 1000
# # squeeze net
# model.to(device)

# training_loss_data = []
# test_loss_data = []
# training_acc_data = []
# test_acc_data = []
# # torch.nn.init.xavier_uniform(model(Linear.weight))

# criterion = torch.nn.CrossEntropyLoss()
# # criterion = torch.nn.BCELoss()
# # criterion = torch.nn.CrossEntropyLoss()
# # optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)
# # optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# for itr in range(max_iters):
#     total_loss = 0
#     total_acc = 0
#     test_total_loss = 0
#     test_total_acc = 0
#     for batch_idx, (x, target) in enumerate(train_loader):
#         # print(x.size())
#         # x = x.reshape(x.size()[0],3,224,224)
#         out =  model(x)
#         # print(out.size())
#         # print(target.size())
#         loss = criterion(out,target)
#         optimizer.zero_grad()
#         loss.backward()

#         optimizer.step()
#         _, predicted = torch.max(out.data, 1)

#         total_acc += ((target==predicted).sum().item())
#         # print(total_acc)
#         total_loss+=loss
#     for batch_idx, (x, target) in enumerate(test_loader):
#         # print(x.size())
#         # x = x.reshape(x.size()[0],3,224,224)
#         out =  model(x)
#         # print(out.size())
#         # print(target.size())
#         test_loss = criterion(out,target)
#         optimizer.zero_grad()
#         test_loss.backward()

#         optimizer.step()
#         _, predicted = torch.max(out.data, 1)

#         test_total_acc += ((target==predicted).sum().item())
#         # print(total_acc)
#         test_total_loss+=loss
#     ave_acc = total_acc/examples
#     test_ave_acc = test_total_acc/test_examples
#     print('total loss: ' + str(total_loss))
#     print('accuracy: ' + str(ave_acc))
#     print('test loss: ' + str(test_total_loss))
#     print('test accuracy: ' + str(test_ave_acc))
#     training_loss_data.append(total_loss / (examples / batch_size))
#     training_acc_data.append(ave_acc)
#     test_loss_data.append(test_total_loss / (test_examples / batch_size))
#     test_acc_data.append(test_ave_acc)
# plt.figure(0)
# plt.plot(np.arange(max_iters), training_loss_data, 'r')
# plt.plot(np.arange(max_iters), test_loss_data, 'b')
# plt.legend(['training loss','valid loss'])

# plt.show()
# plt.figure(1)
# plt.plot(np.arange(max_iters),training_acc_data,'r')
# plt.plot(np.arange(max_iters),test_acc_data,'b')
# plt.legend(['training accuracy','valid accuracy'])

# plt.show()



import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights


# https://www.robots.ox.ac.uk/~vgg/data/flowers/17/
# https://cs231n.github.io/transfer-learning/
# SOURCE: https://pytorch.org/hub/pytorch_vision_squeezenet/
train_dataset = datasets.ImageFolder(root='../data/oxford-flowers17/train', 
                                     transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ]))
valid_dataset = datasets.ImageFolder(root='../data/oxford-flowers17/val', 
                                     transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ]))
train_dataset_size = len(train_dataset)
valid_dataset_size = len(valid_dataset)
print(valid_dataset_size)
# print(train_dataset[0])
# plt.imshow(train_dataset[0][0])
# plt.show()

batch_size = 30
num_batches = train_dataset_size // batch_size
train_dataset_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
valid_dataset_loader = torch.utils.data.DataLoader(valid_dataset, shuffle=False, batch_size=1)
    
class ScratchNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, 5, 1)
        self.conv2 = torch.nn.Conv2d(10, 50, 5, 1)
        self.conv3 = torch.nn.Conv2d(50, 200, 5, 1)
        self.fc1 = torch.nn.Linear(24*24*200, 512) # was 4*4*50, 500 for
        self.fc2 = torch.nn.Linear(512, 17) # was 500,10 for mnist

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        # print(x.size())

        x = x.view(-1, 24*24*200)
        # print(x.size())
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

max_iter = 20
learning_rate = 1e-3
model = ScratchNet()

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4, nesterov=True, momentum=0.9)

train_loss = []
train_loss_overall = []
train_acc = []
train_acc_overall = []
valid_loss = []
valid_acc = []
for epoch in range(max_iter + 1):

    cum_valid_correct = 0
    cum_valid_loss = 0
    with torch.no_grad():
        model.eval()
        for x, y in valid_dataset_loader:
            y_probs = model.forward(x)
            _, y_pred = torch.max(y_probs, dim=1)
            cum_valid_correct += torch.eq(y_pred, y).sum().item()
            loss = loss_function(y_probs, y)
            cum_valid_loss += loss.item() # loss is tensor
    valid_loss.append(cum_valid_loss / valid_dataset_size)
    valid_acc.append(cum_valid_correct / valid_dataset_size)
    print(f"Epoch {epoch} valid: {cum_valid_loss/valid_dataset_size} {cum_valid_correct/valid_dataset_size}")

    if epoch == max_iter: break

    cum_loss = 0
    cum_correct = 0
    model.train()
    for i, (x, y) in enumerate(train_dataset_loader):
        optimizer.zero_grad()
        y_probs = model.forward(x)
        _, y_pred = torch.max(y_probs, dim=1)
        loss = loss_function(y_probs, y)
        loss.backward()
        optimizer.step()

        correct = torch.eq(y_pred, y).sum().item()
        cum_correct += correct
        cum_loss += loss.item() * batch_size# loss is tensor
        train_loss.append(loss.item())
        train_acc.append(correct / batch_size)
        if i % (num_batches // 6) == 0:
            print(f"Epoch {epoch} batch {i} loss: {cum_loss} acc: {cum_correct}")


    train_loss_overall.append(cum_loss / train_dataset_size)
    train_acc_overall.append(cum_correct / train_dataset_size)
    print(f"Epoch {epoch} loss: {cum_loss} acc:{cum_correct/train_dataset_size}")

fig, ax = plt.subplots(1, 2)
ax[0].plot(np.arange(len(train_loss))/num_batches, train_loss, 'co', markersize=1)
ax[0].plot(np.arange(1, max_iter+1), train_loss_overall, 'b-')
ax[0].plot(np.arange(0, max_iter+1), valid_loss, 'r-')
ax[0].legend(['Training Minibatches', 'Training', 'Validation'])
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('average loss')
ax[0].set_title("Average loss")
ax[1].plot(np.arange(len(train_acc))/num_batches, train_acc, 'co', markersize=1)
ax[1].plot(np.arange(1, max_iter+1), train_acc_overall, 'b-')
ax[1].plot(np.arange(0, max_iter+1), valid_acc, 'r-')
ax[1].legend(['Training Minibatches', 'Training', 'Validation'])
ax[1].set_xlabel('epoch')
ax[1].set_ylabel('accuracy')
ax[1].set_title("Classification accuracy")
plt.show()