import torch
import numpy as np

def get_random_batches(x,y,batch_size):
    indices = torch.randperm(x.shape[0])
    batch_ind = [indices[batch_size*i:batch_size*i+batch_size]
         for i in range(x.shape[0]//batch_size)]
    batches = [(x[ind], y[ind]) for ind in batch_ind]
    return batches

def initialize_weights(in_size, out_size, device):
    W = torch.normal(0, np.sqrt(2 / (in_size + out_size)), 
        size=(in_size, out_size),
        requires_grad=True,
        device=device)
    b = torch.zeros(out_size, requires_grad=True, device=device)
    return W, b

def forward(h_prev, W, b, activation):
    a = h_prev.mm(W) + b
    h = activation(a)
    return h
    
def compute_loss_and_acc(y, probs):   
    # calculate accuracy
    y_np = y.numpy()
    probs_np = probs.detach().numpy()
    y_hat = (probs_np == np.vstack(np.max(probs_np, axis=1)))
    match = (y_hat == y_np)
    matches = np.count_nonzero(np.all(match, axis=1))
    instances = match.shape[0]
    acc = matches / instances

    # calculate loss
    loss_one = -torch.sum(y * torch.log(probs), dim=1)
    loss = torch.sum(loss_one)

    return loss, acc 

def sigmoid(x):
    res = 1 / (1 + torch.exp(-x))
    return res

def softmax(x):
    # no stability trick is performed here
    x_exp = torch.exp(x)
    x_exp_sum = torch.sum(x_exp, dim=1)
    res = x_exp / x_exp_sum[:, None]
    return res
