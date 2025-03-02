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
# print(train_dataset[0])
# plt.imshow(train_dataset[0][0])
# plt.show()

batch_size = 20
num_batches = train_dataset_size // batch_size
train_dataset_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
valid_dataset_loader = torch.utils.data.DataLoader(valid_dataset, shuffle=True, batch_size=1)

class SqueezeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = squeezenet1_1(weights=SqueezeNet1_1_Weights).features
        self.classifier = nn.Sequential( # adapted from pytorch source code
            nn.Conv2d(512, 17, kernel_size=1), 
            nn.ReLU(inplace=True), 
            nn.AdaptiveAvgPool2d((1, 1)),
            # nn.Softmax(),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.features(x)
        # x = x.flatten()
        x = self.classifier(x)
        return x
    
max_iter = 20
learning_rate_classifier = 5e-3
learning_rate_features = 1e-4
model = SqueezeNet()

loss_function = nn.CrossEntropyLoss()
optimizer_classifier = torch.optim.SGD(model.classifier.parameters(), lr=learning_rate_classifier)
optimizer_features = torch.optim.SGD(model.features.parameters(), lr=learning_rate_features)

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
    for i, (x, y) in enumerate(train_dataset_loader):
        optimizer_classifier.zero_grad()
        optimizer_features.zero_grad()
        y_probs = model.forward(x)
        _, y_pred = torch.max(y_probs, dim=1)
        loss = loss_function(y_probs, y)
        loss.backward()
        optimizer_classifier.step()
        optimizer_features.step()

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