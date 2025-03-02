import numpy as np
import cv2
from BRIEF import briefLite, briefMatch, plotMatches

def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    #############################
    # TO DO ...
    num_corr = p1.shape[1] # number of correspondences
    u = np.hstack((p2.T, np.ones((num_corr, 1))))
    A = np.vstack((
        np.hstack((u, np.zeros((num_corr, 3)), - p1[[0], :].T * u)),
        np.hstack((np.zeros((num_corr, 3)), u, - p1[[1], :].T * u))
    ))
    U, S, Vh = np.linalg.svd(A)
    h = Vh[-1, :]
    H2to1 = h.reshape((3, 3))
    # print(h, H2to1)
    return H2to1

def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    ''' 
    ###########################
    # TO DO ...

    def applyH(H, locs2):
        transformed_locs2 = H @ np.vstack((locs2, np.ones((1, locs2.shape[1])))) 
        return (transformed_locs2[:2, :].T / transformed_locs2[[2], :].T).T
    
    all_locs1 = locs1[matches[:, 0], :2].T
    all_locs2 = locs2[matches[:, 1], :2].T
    max_num_inliers = 0
    max_is_inliers = None

    for _ in range(num_iter):

        sample_matches = np.random.randint(0, matches.shape[0]-1, size=(4,))

        match_locs1 = all_locs1[:, sample_matches]
        match_locs2 = all_locs2[:, sample_matches]

        H = computeH(match_locs1, match_locs2) # H = cv2.getPerspectiveTransform(match_locs2.T.astype(np.float32), match_locs1.T.astype(np.float32))

        transformed_locs2 = applyH(H, all_locs2) # apply the matrix back to ALL points

        # find inliers: pixel distance < tol
        dist = np.sqrt(np.sum((transformed_locs2 - all_locs1) ** 2, axis=0))
        is_inlier = dist < tol
        num_inliers = np.count_nonzero(is_inlier)
        # inliers = np.where(is_inlier)[0]
        
        if (num_inliers > max_num_inliers):
            max_num_inliers = num_inliers
            max_is_inliers = is_inlier

    # Calculate the bestH based on the results
    inliers = np.where(max_is_inliers)[0]
    inlier_locs1 = all_locs1[:, inliers]
    inlier_locs2 = all_locs2[:, inliers]
    bestH = computeH(inlier_locs1, inlier_locs2)

    # # plot the results
    # im1 = cv2.imread('../data/incline_L.png')
    # im2 = cv2.imread('../data/incline_R.png')

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(2, 2)
    # ax[0, 0].imshow(im1)
    # ax[0, 0].plot(inlier_locs1[0, :], inlier_locs1[1, :], 'rx')
    # ax[0, 1].imshow(im2)
    # ax[0, 1].plot(inlier_locs2[0, :], inlier_locs2[1, :], 'rx')    

    # im2_transformed = cv2.warpPerspective(im2, bestH, dsize=im1.shape[-2::-1])
    # transformed_locs2 = bestH @ np.vstack((inlier_locs2, np.ones((1, inlier_locs2.shape[1])))) 
    # transformed_locs2 = (transformed_locs2[:2, :].T / transformed_locs2[[2], :].T).T

    # ax[1, 0].imshow(im2_transformed)

    # ax[1, 1].imshow(im2_transformed)
    # ax[1, 1].plot(inlier_locs1[0, :], inlier_locs1[1, :], 'bo')  
    # ax[1, 1].plot(transformed_locs2[0, :], transformed_locs2[1, :], 'gx')
    # ax[1, 1].set_xlim(0, im1.shape[1])
    # ax[1, 1].set_ylim(im1.shape[0], 0)
    # plt.show()

    return bestH
        
    
if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)

    # if you don't want to calculate the matches every time
    # try:
    #     with open("planarHpoints.npy", "rb") as f:
    #         locs1 = np.load(f)
    #         locs2 = np.load(f)
    #         matches = np.load(f)
    # except:
    #     locs1, desc1 = briefLite(im1)
    #     locs2, desc2 = briefLite(im2)
    #     matches = briefMatch(desc1, desc2)
    #     with open("planarHpoints.npy", "wb") as f:
    #         np.save(f, locs1)
    #         np.save(f, locs2)
    #         np.save(f, matches)
    
    ransacH(matches, locs1, locs2, num_iter=5000)

