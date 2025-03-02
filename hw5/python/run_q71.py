import torch
import torch.nn
import numpy as np
import scipy.io
from nnq7 import *
import matplotlib.pyplot as plt

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
train_x = torch.from_numpy(train_x).to(torch.float32)
train_y = torch.from_numpy(train_y).to(torch.float32)
valid_x = torch.from_numpy(valid_x).to(torch.float32)
valid_y = torch.from_numpy(valid_y).to(torch.float32)

device = torch.device('cpu')

# Using same hyperparameters from 
max_iters = 150
# pick a batch size, learning rate
batch_size = 50
learning_rate = 8e-4

input_size = 1024
hidden_size = 64
output_size = 36

# create batches
batches = get_random_batches(train_x, train_y, batch_size)
batch_num = len(batches)

# weight initialization
Wlayer1, blayer1 = initialize_weights(input_size, hidden_size, device)
Woutput, boutput = initialize_weights(hidden_size, output_size, device)

# with default settings, you should get loss < 150 and accuracy > 80%
train_loss, train_acc = [], []
valid_loss, valid_acc = [], []
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb, yb in batches:
        
        # forward
        hlayer1 = forward(xb, Wlayer1, blayer1, sigmoid)
        probs = forward(hlayer1, Woutput, boutput, softmax)

        # loss
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        total_acc += acc / batch_num

        # backward
        loss.backward()

        with torch.no_grad():
            # Checking
            # gradient = probs - yb
            # grad_W = hlayer1.transpose(0, 1).mm(gradient)            # (BS * 1)
            # grad_b = torch.sum(gradient, dim=0)
            # print(grad_b, boutput.grad)
            # input()
            Wlayer1 -= learning_rate * Wlayer1.grad
            blayer1 -= learning_rate * blayer1.grad
            Woutput -= learning_rate * Woutput.grad
            boutput -= learning_rate * boutput.grad
            Wlayer1.grad.zero_()
            blayer1.grad.zero_()
            Woutput.grad.zero_()
            boutput.grad.zero_()

    train_loss.append(total_loss.detach().numpy())
    train_acc.append(total_acc)

    hlayer1 = forward(valid_x, Wlayer1, blayer1, sigmoid)
    probs = forward(hlayer1, Woutput, boutput, softmax)
    loss, acc = compute_loss_and_acc(valid_y, probs)
    valid_loss.append(loss.detach().numpy())
    valid_acc.append(acc)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(
            itr, total_loss, total_acc))

# run on validation set and report accuracy! should be above 75%
print('Training Accuracy: ', train_acc[-1])
print('Validation accuracy: ', valid_acc[-1])

# Q3.1.2
fig, ax = plt.subplots(1, 2)
ax[0].plot(np.arange(max_iters), train_acc, 'b-')
ax[0].plot(np.arange(max_iters), valid_acc, 'r-')
ax[0].legend(['Training', 'Validation'])
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('accuracy')
ax[0].set_title("Classification accuracy")
ax[1].plot(np.arange(max_iters), np.array(train_loss)/train_x.shape[0], 'c-')
ax[1].plot(np.arange(max_iters), np.array(valid_loss)/valid_x.shape[0], 'm-')
ax[1].legend(['Training', 'Validation'])
ax[1].set_xlabel('epoch')
ax[1].set_ylabel('average loss')
ax[1].set_title("Average loss")
plt.show()
