from BRIEF import briefLite, briefMatch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def plotMatches(im1, im2, matches, locs1, locs2, locs2_rectified):
    fig = plt.figure()  
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])  
    im = np.zeros((imH, im1.shape[1]+im2.shape[1]), dtype='uint8')  
    im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)  
    im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)  
    plt.imshow(im, cmap='gray')  
    for i in range(matches.shape[0]):
        pt1 = locs1[matches[i,0], 0:2]  
        pt2 = locs2[matches[i,1], 0:2].copy()  
        pt2[0] += im1.shape[1]  
        x = np.asarray([pt1[0], pt2[0]])  
        y = np.asarray([pt1[1], pt2[1]])  
        plt.plot(x,y,'r',linewidth=0.5)  
        plt.plot(x,y,'g.')  
        pt3 = locs2_rectified[matches[i,1], :].copy()  
        plt.plot(pt3[0], pt3[1], 'b.')
        plt.plot([pt1[0], pt3[0]], [pt1[1], pt3[1]], 'y', linewidth=0.6)  
    plt.show()  
    

im = cv2.imread('../data/model_chickenbroth.jpg')
im_sqr = cv2.copyMakeBorder(im, 0, 0, (im.shape[0] - im.shape[1]) // 2, (im.shape[0] - im.shape[1]) // 2,
                            borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
# cv2.imshow('padded image', im_sqr)
rot_angles = list(range(0, 360, 10))
results = []

for rot_angle in rot_angles:
    
    print(rot_angle)
    rot_matrix = cv2.getRotationMatrix2D(
        (im_sqr.shape[1] // 2, im_sqr.shape[0] // 2), angle=rot_angle, scale=1)
    inv_rot_matrix = np.linalg.inv(np.vstack((rot_matrix, [0, 0, 1])))[:2, :]
    # print(inv_rot_matrix == cv2.getRotationMatrix2D(
    #     (im_sqr.shape[1] // 2, im_sqr.shape[0] // 2), angle=-10, scale=1))

    im_rot = cv2.warpAffine(src=im_sqr, M=rot_matrix, dsize=(im_sqr.shape[1], im_sqr.shape[0]))
    # cv2.imshow('rot', im_rot)
    # cv2.imshow('inv_rot', cv2.warpAffine(src=im_rot, M=inv_rot_matrix, dsize=(im_sqr.shape[1], im_sqr.shape[0])))

    locs1, desc1 = briefLite(im_sqr)
    locs2, desc2 = briefLite(im_rot)
    matches = briefMatch(desc1, desc2)

    # rectify the points
    locs2_rectified = locs2.copy()
    locs2_rectified[:,2] = np.ones(locs2.shape[0])
    locs2_rectified.shape += (1, )
    locs2_rectified = np.matmul(inv_rot_matrix, locs2_rectified)
    locs2_rectified = np.round(locs2_rectified).astype(int).reshape(locs2_rectified.shape[:2])

    loc_diff = locs1[matches[:, 0], :2] - locs2_rectified[matches[:, 1], :]
    manhattan_dist = np.abs(loc_diff[:, 0]) + np.abs(loc_diff[:, 1])
    success = np.count_nonzero(manhattan_dist < 5) # threshold of 5
    total = manhattan_dist.shape[0]
    results.append(success)
    # print(manhattan_dist)
    # plotMatches(im_sqr, im_rot, matches, locs1, locs2, locs2_rectified)

    # print(matches)
    # print(locs2_rectified)

# results = [3 * r for r in rot_angles]
fig = plt.figure()
plt.bar(rot_angles, results, color="red", width=7)
plt.xlabel('angle of rotation (in degrees)')
plt.ylabel('number of correct matches')
plt.title('Angle of rotation against success rate of BRIEF feature matching')
plt.show()