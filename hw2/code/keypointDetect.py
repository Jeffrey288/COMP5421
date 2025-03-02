import numpy as np
import cv2

def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()

def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    DoG_pyramid = []
    ################
    # TO DO ...
    # compute DoG_pyramid here
    gaussian_pyramid = np.split(gaussian_pyramid, gaussian_pyramid.shape[2], axis=2)
    gaussian_pyramid = list(map(lambda x: x[:,:,0], gaussian_pyramid))
    for i in range(len(levels) - 1):
        DoG_pyramid.append(gaussian_pyramid[i+1] - gaussian_pyramid[i])
    DoG_pyramid = np.stack(DoG_pyramid, axis=-1)
    # print(DoG_pyramid)

    DoG_levels = levels[1:]
    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    principal_curvature = None
    ##################
    # TO DO ...
    # Compute principal curvature here
    DoG_pyramid = np.split(DoG_pyramid, DoG_pyramid.shape[2], axis=2)
    DoG_pyramid = list(map(lambda x: x[:,:,0], DoG_pyramid))
    principal_curvature = []
    for layer in DoG_pyramid:
        Dx = cv2.Sobel(layer, cv2.CV_32F, 1, 0) # delD/delx
        Dy = cv2.Sobel(layer, cv2.CV_32F, 0, 1) # delD/dely, etc.
        Dxx = cv2.Sobel(Dx, cv2.CV_32F, 1, 0)
        Dxy = cv2.Sobel(Dy, cv2.CV_32F, 1, 0)
        Dyx = cv2.Sobel(Dx, cv2.CV_32F, 0, 1)
        Dyy = cv2.Sobel(Dy, cv2.CV_32F, 0, 1)
        trace = Dxx + Dyy
        determinant = Dxx * Dyy - Dxy * Dyx
        principal_curvature.append((trace ** 2) / (determinant + 1e-10))
    principal_curvature = np.stack(principal_curvature, axis=-1)

    return principal_curvature

def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,   
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = None
    ##############
    #  TO DO ...
    # Compute locsDoG here
    # we want global extrema
    imH, imW, levels = DoG_pyramid.shape
    inlayer_pad = np.pad(DoG_pyramid, ((1, 1), (1, 1), (0, 0)), 'symmetric')
    exlayer_pad = np.pad(DoG_pyramid, ((0, 0), (0, 0), (1, 1)), 'symmetric')
    neighbours = np.array([inlayer_pad[i:i+imH, j:j+imW, :] for i in range(3) for j in range(3)] + [exlayer_pad[:, :, i:i+levels] for i in range(3)])
    max_candidates = DoG_pyramid >= np.max(neighbours, axis=0)
    min_candidates = DoG_pyramid <= np.min(neighbours, axis=0)
    extr_candidates = np.logical_or(max_candidates, min_candidates)
    edge_surpress = np.abs(principal_curvature) <= th_r
    contrast_thresh = np.abs(DoG_pyramid) >= th_contrast
    candidates = np.logical_and.reduce((extr_candidates, edge_surpress, contrast_thresh))
    locsDoG = np.where(candidates)
    locsDoG = np.stack((locsDoG[1], locsDoG[0], locsDoG[2]), axis=-1)
    # print(locsDoG.shape)

    # old inefficient code
    # def isLocalExtrema(coords):
    #     x, y, d = coords
    #     if (principal_curvature[y, x, d] > th_r) \
    #             or abs(DoG_pyramid[y, x, d]) < th_contrast:
    #         return False
    #     neighbours = np.reshape(DoG_pyramid[(y-1) if 0<=(y-1) else y:((y+1) if (y+1)<shape_y else y)+1, 
    #                 (x-1) if 0<=(x-1) else x:((x+1) if (x+1)<shape_x else x)+1, d], (-1, ))
    #     if (d != shape_d - 1): neighbours = np.hstack((neighbours, DoG_pyramid[y, x, d+1]))
    #     if (d != 0): neighbours = np.hstack((neighbours, DoG_pyramid[y, x, d-1]))
    #     return (DoG_pyramid[y, x, d] == np.min(neighbours) or DoG_pyramid[y, x, d] == np.max(neighbours))
    # print("ayy")
    # shape_y, shape_x, shape_d = DoG_pyramid.shape
    # my, mx, md = np.meshgrid(np.arange(shape_y), np.arange(shape_x), np.arange(shape_d))
    # print("bbeee")
    # candidates = np.stack([mx.ravel(), my.ravel(), md.ravel()], axis=-1)
    # locsDoG = candidates[list(map(isLocalExtrema, candidates))]
    # print(locsDoG.shape)
    return locsDoG
    
def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    ##########################
    # TO DO ....
    # compupte gauss_pyramid, gauss_pyramid here
    gauss_pyramid = createGaussianPyramid(im, sigma0=sigma0, k=k, levels=levels)
    DoG_pyr, DoG_levels = createDoGPyramid(gauss_pyramid, levels=levels)
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    return locsDoG, gauss_pyramid

if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    im_pyr = createGaussianPyramid(im)
    displayPyramid(im_pyr)
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    displayPyramid(DoG_pyr)
    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    # displayPyramid(pc_curvature)
    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)

    # display keypoints using matlab
    import matplotlib.pyplot as plt
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.plot(locsDoG[:, 0], locsDoG[:, 1], "ro", markersize=3)
    plt.show()

