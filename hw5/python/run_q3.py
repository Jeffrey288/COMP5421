import string
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import pickle
import numpy as np
import scipy.io
from nn import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 150
# pick a batch size, learning rate
batch_size = 50
learning_rate = 8e-4
hidden_size = 64

batches = get_random_batches(train_x, train_y, batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
initialize_weights(1024, hidden_size, params, 'layer1')
initialize_weights(hidden_size, 36, params, 'output')
initial_hidden_weights = params['Wlayer1'].copy()

# with default settings, you should get loss < 150 and accuracy > 80%
train_loss, train_acc = [], []
valid_loss, valid_acc = [], []
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb, yb in batches:
        
        # if True:  # view the data
        #     for crop in xb:
        #         print(crop)
        #         import matplotlib.pyplot as plt
        #         plt.imshow(crop.reshape(32, 32).T, cmap="gray")
        #         plt.show()

        # training loop can be exactly the same as q2!
        # forward
        h1 = forward(xb, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)

        # loss
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        total_acc += acc / batch_num

        # backward
        delta2 = backwards(probs - yb, params, 'output', linear_deriv)
        backwards(delta2, params, 'layer1', sigmoid_deriv)

        # apply gradient
        for k, v in sorted(list(params.items())):
            if 'grad' in k:
                name = k.split('_')[1]
                params[name] -= learning_rate * params[k]

    train_loss.append(total_loss)
    train_acc.append(total_acc)

    h1 = forward(valid_x, params, 'layer1')
    probs = forward(h1, params, 'output', softmax)
    loss, acc = compute_loss_and_acc(valid_y, probs)
    valid_loss.append(loss)
    valid_acc.append(acc)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(
            itr, total_loss, total_acc))

# run on validation set and report accuracy! should be above 75%
print('Training Accuracy: ', train_acc[-1])
print('Validation accuracy: ', valid_acc[-1])
# test set:
test_data = scipy.io.loadmat('../data/nist36_test.mat')
test_x, test_y = test_data['test_data'], test_data['test_labels']
h1 = forward(test_x, params, 'layer1')
test_probs = forward(h1, params, 'output', softmax)
loss, acc = compute_loss_and_acc(test_y, test_probs)
print('Test Accuracy: ', acc)

saved_params = {k: v for k, v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

# Q3.1.3
fig = plt.figure()
grid = ImageGrid(fig, 111, (8, 8))
reshaped_initial = [np.reshape(initial_hidden_weights[:, i], (32, 32)) for i in range(hidden_size)]
for i in range(64):
    grid[i].imshow(reshaped_initial[i])
plt.show()

fig = plt.figure()
grid = ImageGrid(fig, 111, (8, 8))
reshaped_trained = [np.reshape(params['Wlayer1'][:, i], (32, 32)) for i in range(hidden_size)]
for i in range(64):
    grid[i].imshow(reshaped_trained[i])
plt.show()

# Q3.1.3

# we use test data for plotting here
confusion_matrix = np.zeros((train_y.shape[1], train_y.shape[1]))
test_yhat = (test_probs == np.vstack(np.max(test_probs, axis=1)))
confusion_matrix = test_y.T @ test_yhat

plt.imshow(confusion_matrix, interpolation='nearest')
plt.grid(True)
plt.xlabel('predicted class')
plt.ylabel('actual class')
plt.xticks(np.arange(36),
           string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),
           string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()
