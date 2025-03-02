import numpy as np
import scipy.io
from nn import *
from collections import Counter

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# initialize layers here
initialize_weights(1024, 32, params, 'layer1')
initialize_weights(32, 32, params, 'hidden')
initialize_weights(32, 32, params, 'hidden2')
initialize_weights(32, 1024, params, 'output')

# initializing momentum
params['m_layer1'] = 0
params['m_hidden'] = 0
params['m_hidden2'] = 0
params['m_output'] = 0

# should look like your previous training loops
train_losses = []
valid_losses = []
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        pass
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions
        
         # forward
        h1 = forward(xb,params,'layer1',relu)
        h2 = forward(h1,params,'hidden',relu)
        h3 = forward(h2,params,'hidden2',relu)
        out = forward(h3,params,'output',sigmoid)   

        total_loss += np.sum((xb - out) ** 2)

        # backward
        delta3 = backwards(-2*(xb - out), params, 'output', sigmoid_deriv)
        delta2 = backwards(delta3, params, 'hidden2', relu_deriv)
        delta1 = backwards(delta2, params, 'hidden', relu_deriv)
        backwards(delta1, params, 'layer1', relu_deriv)

        # apply gradient
        for k, v in sorted(list(params.items())):
            if 'grad' in k:
                name = k.split('_')[1]
                params['m_' + name] = 0.9 * params['m_' + name] - learning_rate * params['grad_' + name]
                params[name] = params[name] + params['m_' + name]

    train_losses.append(total_loss / len(train_x))

    h1 = forward(valid_x,params,'layer1',relu)
    h2 = forward(h1,params,'hidden',relu)
    h3 = forward(h2,params,'hidden2',relu)
    out = forward(h3,params,'output',sigmoid)   
    valid_losses.append(np.sum((valid_x - out) ** 2) / len(valid_x))

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

# visualize some results
import matplotlib.pyplot as plt
# Q5.2
fig = plt.figure()
plt.plot(np.arange(max_iters), train_losses, 'b-')
plt.plot(np.arange(max_iters), valid_losses, 'r-')
plt.legend(["training", "validation"])
plt.xlabel("epoch")
plt.ylabel("total loss")
plt.show()
plt.savefig('5,2 loss curve.png')

# Q5.3.1
# selected_classes = np.floor(np.random.random_sample(5) * 36).astype(int)
selected_classes = ["8", "D", "P", "I", "R"]
selected_classes = ["ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".index(c) for c in selected_classes]
valid_labels = np.reshape(valid_data['valid_labels'] @ np.vstack(np.arange(36, dtype=int)), (-1, ))
selected_data_ind = []
for character in selected_classes:
    # print(np.where(valid_labels == character))
    selected_data_ind += list(np.where(valid_labels == character)[0][:2])
    # print(selected_data_ind)
xb = valid_x[selected_data_ind, :]

h1 = forward(xb,params,'layer1',relu)
h2 = forward(h1,params,'hidden',relu)
h3 = forward(h2,params,'hidden2',relu)
out = forward(h3,params,'output',sigmoid)
for i in range(10):
    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(xb[i].reshape(32,32).T)
    plt.subplot(2,1,2)
    plt.imshow(out[i].reshape(32,32).T)
    # plt.show()
    plt.savefig(f"5,3,1 img{i}.png")

try:
    from skimage.measure import compare_psnr as psnr
except:
    from skimage.metrics import peak_signal_noise_ratio as psnr

# evaluate PSNR
h1 = forward(valid_x,params,'layer1',relu)
h2 = forward(h1,params,'hidden',relu)
h3 = forward(h2,params,'hidden2',relu)
out = forward(h3,params,'output',sigmoid) 
total_psnr = 0  
for i in range(valid_x.shape[0]):
    total_psnr += psnr(valid_x[i, :], out[i, :])
print('Average psnr:', total_psnr/valid_x.shape[0])
# Q5.3.2

