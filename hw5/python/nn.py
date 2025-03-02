import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

# Q 2.1
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    # weights between two layers
    var = 2 / (in_size + out_size)
    # W: (in_size, out_size)
    # b: (out_size)
    b = np.zeros(out_size)
    W = np.random.normal(scale=np.sqrt(var), size=(in_size, out_size))
    
    params['W' + name] = W
    params['b' + name] = b

# Q 2.2.1
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = 1 / (1 + np.exp(-x))
    return res

# Q 2.2.2
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    # your code here
    # pre_act, post_act: (out_size)
    # X: (examples, in_size)
    # W: (in_size, out_size)
    # b: (out_size)
    pre_act = X @ W + b
    post_act = activation(pre_act)    

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

# Q 2.2.2 
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = None
    x_stable = x - np.vstack(np.max(x, axis=1))
    x_exp = np.exp(x_stable)
    x_exp_sum = np.sum(x_exp, axis=1)
    res = x_exp / np.vstack(x_exp_sum)
    return res

# Q 2.2.3
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None
    
    # calculate accuracy
    y_hat = (probs == np.vstack(np.max(probs, axis=1)))
    match = (y_hat == y)
    matches = np.count_nonzero(np.all(match, axis=1))
    instances = match.shape[0]
    acc = matches / instances

    # f(x) just means the output of the network, probs!
    # calculate loss
    loss_one = -np.sum(y * np.log(probs), axis=1)
    loss = np.sum(loss_one)

    # print(y_hat, probs, y, match)
    return loss, acc 

# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]
    # your code here
    # do the derivative through activation first
    # then compute the derivative W,b, and X

    # BS * out_size
    gradient = delta * activation_deriv(post_act)

    # (in_size * BS) * (BS * out_size)
    grad_W = X.T @ gradient

    # (BS * out_size) * (out_size * in_size)
    grad_X = gradient @ W.T

    # (BS * 1)
    grad_b = np.sum(gradient, axis=0)

    # note that for batch updating, we sum the individual
    # gradients for each sample

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

# Q 2.4
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    batch_ind = [indices[batch_size*i:batch_size*i+batch_size]
         for i in range(x.shape[0]//batch_size)]
    batches = [(x[ind], y[ind]) for ind in batch_ind]
    return batches
