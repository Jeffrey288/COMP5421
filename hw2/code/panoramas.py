import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches

def blendWarps(im1, im2, warp1, warp2, matrix1, matrix2, new_im_size):

    # My initial attempt -------------------------------
    # BECAUSE I DIDN"T KNOW THERE WAS A SECTION 
    # TEACHING ME ABOUT IMAGE BLENDINGGGGGGGGGGGGGGGGGGGG

    # # alpha blending
    # im1_mask = cv2.warpPerspective(np.ones(im1.shape[:2], dtype=np.uint8), matrix1, dsize=new_im_size, flags=cv2.INTER_NEAREST)\
    #     .reshape(new_im_size[::-1] + (1, ))
    # im2_mask = cv2.warpPerspective(np.ones(im2.shape[:2], dtype=np.uint8), matrix2, dsize=new_im_size, flags=cv2.INTER_NEAREST)\
    #     .reshape(new_im_size[::-1] + (1, ))
    # blend_mask = im1_mask * im2_mask
    # im1_blend_mask = (1 - im1_mask) | blend_mask
    # im2_blend_mask = (1 - im2_mask) | blend_mask

    # # dist1 = cv2.distanceTransform(im1_blend_mask[:, :, 0], cv2.DIST_L2, 3).reshape(new_im_size[::-1] + (1, )) 
    # dist1 = distance_transform_edt(im1_blend_mask)
    # dist2 = distance_transform_edt(im2_blend_mask)
    # crop1 = dist1 * blend_mask
    # crop2 = dist2 * blend_mask
    # blend_alpha1 = (1 - crop1 / np.max(crop1)) * blend_mask
    # blend_alpha2 = (1 - crop2 / np.max(crop2)) * blend_mask
    # blend_sum = blend_alpha1 + blend_alpha2 + 1e-8
    # normalized_blend_alpha1 = blend_alpha1 / blend_sum
    # normalized_blend_alpha2 = blend_alpha2 / blend_sum
    # alpha1 = (im1_mask & ~blend_mask) + normalized_blend_alpha1 * blend_mask
    # alpha2 = (im2_mask & ~blend_mask) + normalized_blend_alpha2 * blend_mask
    # # alpha1 = cv2.GaussianBlur(alpha1, (9, 9), 0.1).reshape(new_im_size[::-1] + (1, ))
    # # alpha2 = cv2.GaussianBlur(alpha2, (9, 9), 0.1).reshape(new_im_size[::-1] + (1, ))
    # pano_im = warp1 * alpha1 + warp2 * alpha2
    # pano_im = pano_im.astype(np.uint8)

    # import matplotlib.pyplot as plt
    # import mplcursors
    # mplcursors.cursor(hover=True)
    # fig, ax = plt.subplots(3, 3)
    # ax[0, 0].imshow(dist1)
    # ax[0, 1].imshow(dist1 * blend_mask)
    # ax[0, 2].imshow(im1_blend_mask)
    # ax[1, 0].imshow(blend_alpha1)
    # ax[1, 1].imshow(alpha1)
    # ax[1, 2].imshow(pano_im)
    # ax[2, 0].imshow(im1_mask)
    # ax[2, 1].imshow(np.exp(blend_alpha1))
    # ax[2, 2].imshow(np.exp(blend_alpha2))
    # plt.show()

    # Implementation accroding to the hints given in the homework ===========================
    mask1 = np.zeros((im1.shape[0], im1.shape[1]))
    mask1[0,:], mask1[-1,:], mask1[:,0], mask1[:,-1] = 1, 1, 1, 1
    mask1 = distance_transform_edt(1-mask1)
    # mask1 = np.nan_to_num(mask1)
    mask1 = mask1/np.max(mask1) # normalization

    mask2 = np.zeros((im2.shape[0], im2.shape[1]))
    mask2[0,:], mask2[-1,:], mask2[:,0], mask2[:,-1] = 1, 1, 1, 1
    mask2 = distance_transform_edt(1-mask2)
    # mask2 = np.nan_to_num(mask2)
    mask2 = mask2/np.max(mask2) # normalization

    im1_mask = cv2.warpPerspective(mask1, matrix1, dsize=new_im_size)
    im2_mask = cv2.warpPerspective(mask2, matrix2, dsize=new_im_size)
    im1_bw = im1_mask > 0
    im2_bw = im2_mask > 0
    intersection = im1_bw & im2_bw
    im1_intersect = intersection * im1_mask
    im2_intersect = intersection * im2_mask
    intersect_sum = im1_intersect + im2_intersect + 1e-8
    im1_weight = im1_intersect / intersect_sum
    im2_weight = im2_intersect / intersect_sum

    pano_im = warp1.copy()
    pano_im[im2_bw, :] = warp2[im2_bw, :]
    pano_im[intersection] = (im1_weight[:, :, np.newaxis] * warp1 + im2_weight[:, :, np.newaxis] * warp2)[intersection]

    # Visual the weights and stuff
    # import matplotlib.pyplot as plt
    # import mplcursors
    # mplcursors.cursor(hover=True)
    # fig, ax = plt.subplots(3, 3)
    # ax[0, 0].imshow(im1_mask)
    # ax[0, 1].imshow(im2_mask)
    # ax[0, 2].imshow(intersection)
    # boundary_points = np.array([
    #     [0, 0],
    #     [0, im2.shape[0]-1],
    #     [im2.shape[1]-1, 0],
    #     [im2.shape[1]-1, im2.shape[0]-1]
    # ]).T
    # transformed_boundary_points = H2to1 @ np.vstack((boundary_points, np.ones((1, boundary_points.shape[1]))))
    # transformed_boundary_points = (transformed_boundary_points[:2, :].T / transformed_boundary_points[[2], :].T).T
    # print(transformed_boundary_points)
    # ax[1, 0].imshow(pano_im)
    # ax[1, 0].plot(transformed_boundary_points[0, :], transformed_boundary_points[1, :], 'bo')
    # # ax[1, 1].imshow(im1_intersect)
    # # ax[1, 2].imshow(im2_intersect)
    # # ax[2, 0].imshow(im1_weight)
    # # ax[2, 1].imshow(im2_weight)
    # ax[1, 1].imshow(im1)
    # ax[1, 2].imshow(im2)
    # ax[2, 0].imshow(warp1)
    # ax[2, 1].imshow(warp2)
    # ax[2, 2].imshow(im1)
    # plt.show()

    return pano_im

def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    #######################################
    # TO DO ...
    boundary_points = np.array([
        [0, 0],
        [0, im2.shape[0]-1],
        [im2.shape[1]-1, 0],
        [im2.shape[1]-1, im2.shape[0]-1]
    ]).T
    transformed_boundary_points = H2to1 @ np.vstack((boundary_points, np.ones((1, boundary_points.shape[1]))))
    transformed_boundary_points = (transformed_boundary_points[:2, :].T / transformed_boundary_points[[2], :].T).T
    new_im_size = tuple(np.max([np.max(transformed_boundary_points, axis=1), im1.shape[1::-1]], axis=0).astype(int))

    im1_out = cv2.warpPerspective(im1, np.eye(3), dsize=new_im_size, flags=cv2.INTER_NEAREST)
    im2_out = cv2.warpPerspective(im2, H2to1, dsize=new_im_size, flags=cv2.INTER_NEAREST)
    pano_im = blendWarps(im1, im2, im1_out, im2_out, np.eye(3), H2to1, new_im_size)

    return pano_im


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    ''' 
    ######################################
    # TO DO ...

    boundary_points = np.array([
        [0, 0],
        [0, im2.shape[0]-1],
        [im2.shape[1]-1, 0],
        [im2.shape[1]-1, im2.shape[0]-1]
    ]).T
    transformed_boundary_points = H2to1 @ np.vstack((boundary_points, np.ones((1, boundary_points.shape[1]))))
    transformed_boundary_points = transformed_boundary_points[:2, :].T / transformed_boundary_points[[2], :].T  
    candidates = np.vstack((transformed_boundary_points, [[0, 0],
                        [0, im1.shape[0]-1],
                        [im1.shape[1]-1, 0],
                        [im1.shape[1]-1, im1.shape[0]-1]])).astype(int)
    # print(candidates)
    min_pts, max_pts = np.min(candidates, axis=0), np.max(candidates, axis=0)
    translate = (-min_pts)
    new_img_size = tuple(max_pts - min_pts)

    M = np.vstack([np.hstack([np.eye(2), np.vstack(translate)]), [[0, 0, 1]]])
    transformed_im1 = cv2.warpAffine(im1, M[:2, :], new_img_size)
    transformed_im2 = cv2.warpPerspective(im2, M @ H2to1, new_img_size)

    pano_im = blendWarps(im1, im2, transformed_im1, transformed_im2, M, M@H2to1, new_img_size)
    return pano_im

def generatePanorama(im1, im2):
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    return imageStitching_noClip(im1, im2, H2to1)
    
if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')

    # print(im1.shape)
    # locs1, desc1 = briefLite(im1)
    # locs2, desc2 = briefLite(im2)
    # matches = briefMatch(desc1, desc2)
    # H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)

    try:
        with open("../temp/panoramaPoints.npy", "rb") as f:
            locs1 = np.load(f)
            locs2 = np.load(f)
            matches = np.load(f)
            H2to1 = np.load(f)
    except:
        locs1, desc1 = briefLite(im1)
        locs2, desc2 = briefLite(im2)
        matches = briefMatch(desc1, desc2)
        H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
        with open("../temp/panoramaPoints.npy", "wb") as f:
            np.save(f, locs1)
            np.save(f, locs2)
            np.save(f, matches)
            np.save(f, H2to1)

    # q6.1
    pano_im = imageStitching(im1, im2, H2to1)
    cv2.imshow('panoramas 6.1', pano_im)
    cv2.waitKey(0)
    cv2.imwrite('../results/6_1.png', pano_im)
    np.save("../results/q6_1.npy", H2to1)

    # q6.2
    pano_im = imageStitching_noClip(im1, im2, H2to1)
    cv2.imshow('panoramas 6.2', pano_im)
    cv2.waitKey(0)
    cv2.imwrite('../results/6_2.png', pano_im)

    # q6.3
    pano_im = generatePanorama(im1, im2)
    cv2.imshow('panoramas 6.3', pano_im)
    cv2.waitKey(0)
    cv2.imwrite('../results/6_3.png', pano_im)

    cv2.destroyAllWindows()