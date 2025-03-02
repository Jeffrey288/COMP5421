import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation
import skimage.transform

from nn import *
from q4 import *

# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

class CNN(nn.Module): # image size is 28x28
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(\
            nn.Conv2d(1, 20, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 16, 5), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(400, 180), nn.ReLU(),
            nn.Linear(180, 100), nn.ReLU(),
            nn.Linear(100, 47)
        )

    def forward(self, x):
        x = self.net(x)
        return x
    
device = torch.device('cpu')

def train_model(cnn=None):
        
    train_data = datasets.EMNIST(root='../data', split="balanced", train=True, transform=ToTensor(), download=True)
    test_data = datasets.EMNIST(root='../data', split="balanced", train=False, transform=ToTensor())
    train_x, train_y = train_data.data, train_data.targets
    test_x, test_y = test_data.data, test_data.targets
    # print(train_x[0].size())
    # plt.imshow(train_x[0], cmap='gray')
    # plt.show()

    num_samples = train_x.size(0)
    print(num_samples)
    batch_size = 200 # note that there are 112800 samples
    num_batches = train_x.size(0) // batch_size
    max_iter = 10
    learning_rate = 0.09

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    if cnn is None:
        cnn = CNN()

    loss_function = nn.CrossEntropyLoss() # better for that
    # stochastic (mini-batch) gradient method 
    gradient_method = optim.SGD(cnn.parameters(), lr=learning_rate, nesterov=True, momentum=0.9)

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
            cum_loss += loss.item() * batch_size# loss is tensor
            train_loss.append(loss.item())
            train_acc.append(correct / batch_size)

            if i % 100 == 0:
                print(f"Epoch {epoch} batch {i} loss: {cum_loss} acc: {cum_correct}")
        
        train_loss_overall.append(cum_loss / train_x.size()[0])
        train_acc_overall.append(cum_correct / train_x.size()[0])
        print(f"Epoch {epoch} loss: {cum_loss} acc:{cum_correct/train_x.size()[0]}")

    # save the model:
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    torch.save(cnn.state_dict(), 'run_q74_model.pickle')
    print("saved model successfully!")

    # # print(train_loss)

    # # print(train_loss)
    # print(train_loss_overall)
    # print(train_acc_overall)
    # print(test_loss)
    # print(test_acc)

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



    return cnn

def load_model():
    cnn = CNN()
    cnn.load_state_dict(torch.load('run_q74_model.pickle'))
    cnn.eval()
    return cnn
 

if __name__ == "__main__":
    # uncomment to train the model
    # cnn = load_model()
    # cnn = train_model(cnn)
    cnn = train_model()
    cnn = load_model()

    # COPIED FROM Q4
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    for img in os.listdir('../images'):
        im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
        bboxes, bw = findLetters(im1)

        plt.imshow(bw)
        for bbox in bboxes:
            minr, minc, maxr, maxc = bbox
            rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                    fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
        plt.show()

        box_no = len(bboxes)
        img_centers_y = np.array([(bbox[0] + bbox[2])/2 for bbox in bboxes])
        img_centers_x = np.array([(bbox[1] + bbox[3])/2 for bbox in bboxes])
        clusters = [[0]]   
        cluster_len = [1]
        centroids = np.array([img_centers_y[0]])

        sorted_ind = np.argsort(img_centers_y)                      
        sorted_y = img_centers_y[sorted_ind]                    # sorted_y 1 3 7 10 28 31 34 39 60 61 62 65 etc.
        diff_y = np.diff(sorted_y)                              # diff_y 2 4 3 18 3 3 21 1 1 3              
        sorted_diff_y = np.sort(diff_y)                         # sorted_diff_y 1 1 2 3 3 3 3 4 18 21
        diff_sorted_diff_y = np.diff(sorted_diff_y)             # diff_sorted_diff_y 0 1 1 0 0 0 0 1 14 3
        thresh = sorted_diff_y[np.argmax(diff_sorted_diff_y)]   # max: 14, sorted_diff_y[argmax(diff_sorted_diff_y)] = 4
        split_points = np.argwhere(diff_y > thresh)[:, 0]           # for now the performance is fine
        
        clusters = np.split(sorted_ind, split_points + 1)
        clusters = [cluster[np.argsort(img_centers_x[cluster])] for cluster in clusters]

        cropped_images = [bw[bbox[0]:bbox[2]+1, bbox[1]:bbox[3]+1] for bbox in bboxes]
        edited_images = []
        for img in cropped_images:
            # pad the images such that they are squares
            diff = abs(img.shape[1] - img.shape[0])
            if img.shape[0] > img.shape[1]:
                padded = np.pad(img, ((0, 0), (diff//2, diff-diff//2)), mode="constant", constant_values=(1.0,))
            elif img.shape[1] > img.shape[0]:
                padded = np.pad(img, ((diff//2, diff-diff//2), (0, 0)), mode="constant", constant_values=(1.0,))
            else:
                padded = img

            pad_size = 2
            resized = skimage.transform.resize(padded, (28-pad_size*2, 28-pad_size*2), anti_aliasing=True)
            res = np.pad(resized, ((pad_size, pad_size), (pad_size, pad_size)), 'constant', constant_values=(1.0, ))
            thresh = skimage.filters.threshold_otsu(res)
            res = (res > thresh).astype(float)
            res = skimage.morphology.dilation(1.0 - res, skimage.morphology.square(2))
            edited_images.append((res.T).reshape((-1, )))
            # plt.imshow(res, cmap="gray")
            # plt.show()

        ### BELOW IS Q7.4 CODE
        ### DO NOT MODIFY
        ### CHANGE 32 to 28 ABOVE
        ### and remove 1.0 - from res
        import string
        letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])

        image_x = np.vstack(edited_images).reshape((-1, 1, 28, 28))
        probs = cnn(torch.from_numpy(image_x).type(torch.FloatTensor)).detach().numpy()
        image_y = (probs == np.vstack(np.max(probs, axis=1))) @ np.vstack(np.arange(47, dtype=int))
        image_y = image_y.reshape((-1, ))
        print(image_y)
        image_letters = ["0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"[c] for c in image_y]
        for cluster in clusters:
            print("".join([image_letters[ind] for ind in cluster]))

    

