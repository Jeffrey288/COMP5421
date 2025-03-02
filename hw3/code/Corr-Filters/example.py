import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import animation
import matplotlib.patches as patches
import scipy.ndimage

img = np.load('lena.npy')

# template cornes in image space [[x1, x2, x3, x4], [y1, y2, y3, y4]]
pts = np.array([[248, 292, 248, 292],
                [252, 252, 280, 280]])

# size of the template (h, w)
dsize = np.array([pts[1, 3] - pts[1, 0] + 1,
                  pts[0, 1] - pts[0, 0] + 1])

# set template corners
tmplt_pts = np.array([[0, dsize[1]-1, 0, dsize[1], -1],
                      [0, 0, dsize[0] - 1, dsize[0] - 1]])


# apply warp p to template region of img
def imwarp(p):
    global img, dsize
    return img[p[1]:(p[1]+dsize[0]), p[0]:(p[0]+dsize[1])]


# get positive example
gnd_p = np.array([252, 248])  # ground truth warp
x = imwarp(gnd_p)  # the template

# stet up figure
fig, axarr = plt.subplots(1, 3)
axarr[0].imshow(img, cmap=plt.get_cmap('gray'))
patch = patches.Rectangle((gnd_p[0], gnd_p[1]), dsize[1], dsize[0],
                          linewidth=1, edgecolor='r', facecolor='none')
axarr[0].add_patch(patch)
axarr[0].set_title('Image')

cropax = axarr[1].imshow(x, cmap=plt.get_cmap('gray'))
axarr[1].set_title('Cropped Image')

dx = np.arange(-np.floor(dsize[1]/2), np.floor(dsize[1]/2)+1, dtype=int)
dy = np.arange(-np.floor(dsize[0]/2), np.floor(dsize[0]/2)+1, dtype=int)
[dpx, dpy] = np.meshgrid(dx, dy)
dpx = dpx.reshape(-1, 1)
dpy = dpy.reshape(-1, 1)
dp = np.hstack((dpx, dpy))
N = dpx.size

all_patches = np.ones((N*dsize[0], dsize[1]))
all_patchax = axarr[2].imshow(all_patches, cmap=plt.get_cmap('gray'),
                              aspect='auto', norm=colors.NoNorm())
axarr[2].set_title('Concatenation of Sub-Images (X)')

X = np.zeros((N, N))
Y = np.zeros((N, 1))

sigma = 5

def q43():
    global X, Y
    reg = 0 # regularization
    g = np.linalg.inv(X @ X.T + reg * np.eye(X.shape[0])) @ X @ Y[np.newaxis, :]
    filt = g.reshape(dsize)
    img_corr = scipy.ndimage.correlate(img, filt)
    fig = plt.figure()
    plt.imshow(img_corr, cmap='gray')
    plt.title(rf'$Correlate, \lambda={reg}$')
    plt.savefig(f'../../writeup/q43_img{reg}.png')
    fig = plt.figure()
    plt.imshow(filt, cmap='gray')
    plt.title(rf'$Filter, \lambda={reg}$')
    plt.savefig(f'../../writeup/q43_filt{reg}.png')
    fig = plt.figure()
    img_conv = scipy.ndimage.convolve(img, filt)
    plt.imshow(img_conv, cmap='gray')
    plt.title(rf'$Convolve, \lambda={reg}$')
    plt.savefig(f'../../writeup/q44_img{reg}.png')
    print(np.all(scipy.ndimage.convolve(img, filt[::-1,::-1])==img_corr))
    print(np.all(scipy.ndimage.convolve(img[::-1,::-1], filt)[::-1,::-1]==img_corr))
    plt.show()

# To skip a long wait:
for i in range(N):
    xn = imwarp(dp[i, :] + gnd_p)
    X[:, i] = xn.reshape(-1)
    Y[i] = np.exp(-np.dot(dp[i, :], dp[i, :])/sigma)
    if i % (N//100) == 0: print(f"{i}/{N}")
q43()
plt.show()

# If you like waiting, then use this :D
def init():
    return [cropax, patch, all_patchax]

def animate(i):
    global X, Y, dp, gnd_p, sigma, all_patches, patch, cropax, all_patchax, N

    if i < N:  # If the animation is still running
        xn = imwarp(dp[i, :] + gnd_p)
        X[:, i] = xn.reshape(-1)
        Y[i] = np.exp(-np.dot(dp[i, :], dp[i, :])/sigma)
        all_patches[(i*dsize[0]):((i+1)*dsize[0]), :] = xn
        cropax.set_data(xn)
        all_patchax.set_data(all_patches.copy())
        all_patchax.autoscale()
        patch.set_xy(dp[i, :] + gnd_p)
        return [cropax, patch, all_patchax]
        # if i % (N//100) == 0: print(f"{i}/{N}")
        # return []
    else:  # Stuff to do after the animation ends
        fig3d = plt.figure()
        ax3d = fig3d.add_subplot(111, projection='3d')
        ax3d.plot_surface(dpx.reshape(dsize), dpy.reshape(dsize),
                          Y.reshape(dsize), cmap=plt.get_cmap('coolwarm'))

        # Place your solution code for question 4.3 here
        q43() # view this function for the code
        plt.show()
        return []


# Start the animation
ani = animation.FuncAnimation(fig, animate, frames=N+1,
                              init_func=init, blit=True,
                              repeat=False, interval=10)
plt.show()
