import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
try:
    from skimage.measure import compare_psnr as psnr
except:
    from skimage.metrics import peak_signal_noise_ratio as psnr

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

dim = 32
# do PCA

mean_train = np.mean(train_x, axis=0)
print(mean_train)
# will save svd into npz file
try:
    svd = np.load('q6_SVD.npz')
    U, S, Vh = svd['U'], svd['S'], svd['Vh']
except Exception as e:
    # print(e)
    U, S, Vh = np.linalg.svd(train_x - mean_train)
    np.savez('q6_SVD.npz', U=U, S=S, Vh=Vh)

# U, S, Vh = np.linalg.svd(train_x - mean_train)

V = Vh.T

# train_y = train_x @ V
# recon = train_y[:, :dim] @ V[:, :dim].T

print(S)
conv_matrix = V[:, :dim] @ V[:, :dim].T
print("shape:", conv_matrix.shape)
print("rank:", np.linalg.matrix_rank(conv_matrix))

# rebuild it
recon = (train_x - mean_train) @ conv_matrix + mean_train

# for i in range(5):
#     plt.subplot(2,1,1)
#     plt.imshow(train_x[i].reshape(32,32).T)
#     plt.subplot(2,1,2)
#     plt.imshow(recon[i].reshape(32,32).T)
#     plt.show()

mean_valid = np.mean(valid_x, axis=0)
recon_valid = (valid_x - mean_valid) @ conv_matrix + mean_valid
total = []
for pred,gt in zip(recon_valid,valid_x):
    total.append(psnr(gt,pred))
print(np.array(total).mean())

# selected_classes = np.floor(np.random.random_sample(5) * 36).astype(int)
selected_classes = ["8", "D", "P", "I", "R"]
selected_classes = ["ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".index(c) for c in selected_classes]
valid_labels = np.reshape(valid_data['valid_labels'] @ np.vstack(np.arange(36, dtype=int)), (-1, ))
selected_data_ind = []
for character in selected_classes:
    selected_data_ind += list(np.where(valid_labels == character)[0][:2])

# build valid dataset
valid_samples = valid_x[selected_data_ind, :]
# recon_valid_samples = valid_samples @ conv_matrix
recon_valid_samples = recon_valid[selected_data_ind, :]
for i in range(10):
    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(valid_samples[i].reshape(32,32).T)
    plt.subplot(2,1,2)
    plt.imshow(recon_valid_samples[i].reshape(32,32).T)
    # plt.show()
    plt.savefig(f"6,2 img{i}.png")
