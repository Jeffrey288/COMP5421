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
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
i=1
for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    # -=-=- display the bounding boxes -=-=-
    plt.figure()
    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    # plt.show()
    plt.savefig(f'4,2 img{i}.png')
    i += 1

    # -=-=- find the rows using..RANSAC, counting, clustering, etc. -=-=-

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
    
    # -=-=- display the clusters boxes -=-=-
    # fig = plt.figure()
    # plt.imshow(bw)
    # for cluster, color in zip(clusters, ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'gray', 'black']):
    #     for i, ind in enumerate(cluster):
    #         bbox = bboxes[ind]
    #         minr, minc, maxr, maxc = bbox
    #         rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
    #                                 fill=False, edgecolor=color, linewidth=1+0.15*i)
    #         plt.gca().add_patch(rect)
    # plt.show()
    
    # -=-=- crop the bounding boxes =-=-
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    # apply threshold
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
        # padded = 1.0 - skimage.morphology.dilation(1.0 - padded, skimage.morphology.square(3))
        # plt.imshow(padded, cmap="gray")
        # plt.show()

        # resize the photos
        pad_size = 2
        resized = skimage.transform.resize(padded, (32-pad_size*2, 32-pad_size*2), anti_aliasing=True)
        res = np.pad(resized, ((pad_size, pad_size), (pad_size, pad_size)), 'constant', constant_values=(1.0, ))
        thresh = skimage.filters.threshold_otsu(res)
        res = (res > thresh).astype(float)
        res = 1.0 - skimage.morphology.dilation(1.0 - res, skimage.morphology.square(3))
        edited_images.append((res.T).reshape((-1, )))
        # plt.imshow(res, cmap="gray")
        # plt.show()

    # -=-=- load the weights -=-=-
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))

    image_x = np.vstack(edited_images)
    h1 = forward(image_x, params, 'layer1')
    probs = forward(h1, params, 'output', softmax)
    image_y = (probs == np.vstack(np.max(probs, axis=1))) @ np.vstack(np.arange(36, dtype=int))
    image_y = image_y.reshape((-1, ))
    print(image_y)
    image_letters = ["ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"[c] for c in image_y]
    for cluster in clusters:
        print("".join([image_letters[ind] for ind in cluster]))

    
