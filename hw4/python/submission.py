"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import helper
import scipy.linalg
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import multiprocessing
import scipy.optimize

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''


def eightpoint(pts1, pts2, M):

    # 0. Normalize points
    T = np.diag([1/M, 1/M, 1])
    npts1 = pts1 @ T[:2, :2]
    npts2 = pts2 @ T[:2, :2]

    # Af = 0, where f is the vector of elements in F
    # Only one equation per point correspondence :D
    A = np.hstack(  # 1. construct A
        (npts1[:, [0]] * npts2[:, [0]],
         npts1[:, [0]] * npts2[:, [1]],
         npts1[:, [0]],
         npts1[:, [1]] * npts2[:, [0]],
         npts1[:, [1]] * npts2[:, [1]],
         npts1[:, [1]],
         npts2[:, [0]],
         npts2[:, [1]],
         np.ones((npts1.shape[0], 1)))
    )
    U, S, Vh = np.linalg.svd(A)  # 2. find SVD A
    f = Vh[-1, :]  # 3. solve least squares
    F = f.reshape((3, 3))

    # 4. Enforcing rank 2 constraints / singularity conditions
    U, S, Vh = np.linalg.svd(F)
    F = U @ np.diag([S[0], S[1], 0]) @ Vh

    # refine by local minimization
    # https://www.cs.cornell.edu/courses/cs664/1997sp/stereo-reg.htm
    F = helper.refineF(F, npts1, npts2)

    # 5. Unnormalize
    F = T.T @ F @ T

    # print(F)
    # input()

    return F


'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''

def sevenpoint(pts1, pts2, M):

    # 0. Normalize points
    npts1 = pts1 / M
    npts2 = pts2 / M

    # Note that A is rank 7
    # Since Af = 0,
    # The solutions for f are the
    # linear combinations of nullspaces of A.
    A = np.hstack(  # 1. construct A
        (npts1[:, [0]] * npts2[:, [0]],
         npts1[:, [0]] * npts2[:, [1]],
         npts1[:, [0]],
         npts1[:, [1]] * npts2[:, [0]],
         npts1[:, [1]] * npts2[:, [1]],
         npts1[:, [1]],
         npts2[:, [0]],
         npts2[:, [1]],
         np.ones((npts1.shape[0], 1)))
    )

    # we can do this by calling null_space
    # null_space = scipy.linalg.null_space(A)
    # f1, f2 = null_space[:, 0], null_space[:, 1]

    # alternatively, we can also use svd for this task
    # the last two vectors in V are the nullspaces
    # this is consistent with least-squares
    U, S, Vh = np.linalg.svd(A)
    f1, f2 = Vh[-1, :], Vh[-2, :]

    F1 = helper.refineF(f1, npts1, npts2)
    F2 = helper.refineF(f2, npts1, npts2)
    f1, f2 = F1.reshape((-1, )), F2.reshape((-1, ))

    # F1, F2 = f1.reshape((3, 3)), f2.reshape((3, 3))

    # Computes the polynomial
    # import sympy
    # alpha = sympy.symbols('a')
    # F1 = sympy.Matrix([f'F1{i}' for i in range(9)]).reshape(3, 3)
    # F2 = sympy.Matrix([f'F2{i}' for i in range(9)]).reshape(3, 3)
    # print(sympy.Poly((alpha * F1 + (1 - alpha) * F2).det(), alpha).all_coeffs())

    # Note that det(F) = 0
    # Let F = alpha * F1 + (1 - alpha) * F2
    # coeffs are the coefficients of the polynomial
    # coeffs = [f1[0]*f1[4]*f1[8] - f1[0]*f1[4]*f2[8] - f1[0]*f1[5]*f1[7] + f1[0]*f1[5]*f2[7] + f1[0]*f1[7]*f2[5] - f1[0]*f1[8]*f2[4] + f1[0]*f2[4]*f2[8] - f1[0]*f2[5]*f2[7] - f1[1]*f1[3]*f1[8] + f1[1]*f1[3]*f2[8] + f1[1]*f1[5]*f1[6] - f1[1]*f1[5]*f2[6] - f1[1]*f1[6]*f2[5] + f1[1]*f1[8]*f2[3] - f1[1]*f2[3]*f2[8] + f1[1]*f2[5]*f2[6] + f1[2]*f1[3]*f1[7] - f1[2]*f1[3]*f2[7] - f1[2]*f1[4]*f1[6] + f1[2]*f1[4]*f2[6] + f1[2]*f1[6]*f2[4] - f1[2]*f1[7]*f2[3] + f1[2]*f2[3]*f2[7] - f1[2]*f2[4]*f2[6] - f1[3]*f1[7]*f2[2] + f1[3]*f1[8]*f2[1] - f1[3]*f2[1]*f2[8] + f1[3]*f2[2]*f2[7] + f1[4]*f1[6]*f2[2] - f1[4]*f1[8]*f2[0] + f1[4]*f2[0]*f2[8] - f1[4]*f2[2]*f2[6] - f1[5]*f1[6]*f2[1] + f1[5]*f1[7]*f2[0] - f1[5]*f2[0]*f2[7] + f1[5]*f2[1]*f2[6] + f1[6]*f2[1]*f2[5] - f1[6]*f2[2]*f2[4] - f1[7]*f2[0]*f2[5] + f1[7]*f2[2]*f2[3] + f1[8]*f2[0]*f2[4] - f1[8]*f2[1]*f2[3] - f2[0]*f2[4]*f2[8] + f2[0]*f2[5]*f2[7] + f2[1]*f2[3]*f2[8] - f2[1]*f2[5]*f2[6] - f2[2]*f2[3]*f2[7] + f2[2]*f2[4]*f2[6], f1[0]*f1[4]*f2[8] - f1[0]*f1[5]*f2[7] - f1[0]*f1[7]*f2[5] + f1[0]*f1[8]*f2[4] - 2*f1[0]*f2[4]*f2[8] + 2*f1[0]*f2[5]*f2[7] - f1[1]*f1[3]*f2[8] + f1[1]*f1[5]*f2[6] + f1[1]*f1[6]*f2[5] - f1[1]*f1[8]*f2[3] + 2*f1[1]*f2[3]*f2[8] - 2*f1[1]*f2[5]*f2[6] + f1[2]*f1[3] *
    #           f2[7] - f1[2]*f1[4]*f2[6] - f1[2]*f1[6]*f2[4] + f1[2]*f1[7]*f2[3] - 2*f1[2]*f2[3]*f2[7] + 2*f1[2]*f2[4]*f2[6] + f1[3]*f1[7]*f2[2] - f1[3]*f1[8]*f2[1] + 2*f1[3]*f2[1]*f2[8] - 2*f1[3]*f2[2]*f2[7] - f1[4]*f1[6]*f2[2] + f1[4]*f1[8]*f2[0] - 2*f1[4]*f2[0]*f2[8] + 2*f1[4]*f2[2]*f2[6] + f1[5]*f1[6]*f2[1] - f1[5]*f1[7]*f2[0] + 2*f1[5]*f2[0]*f2[7] - 2*f1[5]*f2[1]*f2[6] - 2*f1[6]*f2[1]*f2[5] + 2*f1[6]*f2[2]*f2[4] + 2*f1[7]*f2[0]*f2[5] - 2*f1[7]*f2[2]*f2[3] - 2*f1[8]*f2[0]*f2[4] + 2*f1[8]*f2[1]*f2[3] + 3*f2[0]*f2[4]*f2[8] - 3*f2[0]*f2[5]*f2[7] - 3*f2[1]*f2[3]*f2[8] + 3*f2[1]*f2[5]*f2[6] + 3*f2[2]*f2[3]*f2[7] - 3*f2[2]*f2[4]*f2[6], f1[0]*f2[4]*f2[8] - f1[0]*f2[5]*f2[7] - f1[1]*f2[3]*f2[8] + f1[1]*f2[5]*f2[6] + f1[2]*f2[3]*f2[7] - f1[2]*f2[4]*f2[6] - f1[3]*f2[1]*f2[8] + f1[3]*f2[2]*f2[7] + f1[4]*f2[0]*f2[8] - f1[4]*f2[2]*f2[6] - f1[5]*f2[0]*f2[7] + f1[5]*f2[1]*f2[6] + f1[6]*f2[1]*f2[5] - f1[6]*f2[2]*f2[4] - f1[7]*f2[0]*f2[5] + f1[7]*f2[2]*f2[3] + f1[8]*f2[0]*f2[4] - f1[8]*f2[1]*f2[3] - 3*f2[0]*f2[4]*f2[8] + 3*f2[0]*f2[5]*f2[7] + 3*f2[1]*f2[3]*f2[8] - 3*f2[1]*f2[5]*f2[6] - 3*f2[2]*f2[3]*f2[7] + 3*f2[2]*f2[4]*f2[6], f2[0]*f2[4]*f2[8] - f2[0]*f2[5]*f2[7] - f2[1]*f2[3]*f2[8] + f2[1]*f2[5]*f2[6] + f2[2]*f2[3]*f2[7] - f2[2]*f2[4]*f2[6]]
    # roots = np.roots(coeffs)
    # However, this code is actually bad because it will cause heavy numerical instability

    def det(alpha): return np.linalg.det(alpha * F1 + (1 - alpha) * F2)
    coeff_0 = det(0)
    coeff_1 = 2*(det(1)-det(-1))/3 - (det(2)-det(-2))/12
    coeff_2 = (det(1)+det(-1))/2 - coeff_0
    coeff_3 = (det(1)-det(-1))/2 - coeff_1
    coeff = np.array([coeff_3, coeff_2, coeff_1, coeff_0])
    roots = np.roots(coeff)

    F = [alpha * F1 + (1 - alpha) * F2 for alpha in roots]
    F = [f.real for f in F if np.all(f.imag == 0)]

    # refine
    F = [helper.refineF(f, npts1, npts2) for f in F]

    # 5. Unnormalize
    T = np.diag([1/M, 1/M, 1.0])
    F = [T.T @ f @ T for f in F]
    return F


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''


def essentialMatrix(F, K1, K2):
    # F = K2^-T @ E @ K1^-1
    # E = K2^T @ F @ K1
    return (K2.T @ F @ K1)


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''


def triangulate(C1, pts1, C2, pts2):

    # construct all A matrices at once
    Aall = np.array([
        C2[[0], :] - pts2[:, [0]] * C2[[2], :],
        C2[[1], :] - pts2[:, [1]] * C2[[2], :],
        C1[[0], :] - pts1[:, [0]] * C1[[2], :],
        C1[[1], :] - pts1[:, [1]] * C1[[2], :]
    ])

    num_pts = pts1.shape[0]
    threeD_pts = []
    for n in range(num_pts):
        A = Aall[:, n, :]  # access one matrix
        U, S, Vh = np.linalg.svd(A)
        pt = Vh[-1, :]
        pt = pt[:3] / pt[3]
        threeD_pts.append(pt)
    threeD_pts = np.vstack(threeD_pts)

    # calculate the reprojection error
    threeD_homo = np.hstack([threeD_pts, np.ones((num_pts, 1))]).T
    pts1_hat = (C1 @ threeD_homo).T
    pts1_hat = pts1_hat[:, :2] / (pts1_hat[:, [2]] + 1e-100)
    pts2_hat = (C2 @ threeD_homo).T
    pts2_hat = pts2_hat[:, :2] / (pts2_hat[:, [2]] + 1e-100)
    err = np.sum((pts1 - pts1_hat) ** 2 + (pts2 - pts2_hat) ** 2)
    # print(np.hstack((pts1_hat, pts1_hat-pts1)))
    # print(np.hstack((pts2_hat, pts2_hat-pts2)))
    # print((pts1 - pts1_hat) ** 2 + (pts2 - pts2_hat) ** 2)
    # print(err)
    # input()

    return threeD_pts, err


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''


def epipolarCorrespondence(im1, im2, F, x1, y1):

    test_radius = 35  # cannot be too big
    window_half_size = 10
    gaussian_const = 0.0001

    # constructing gaussian weighting matrix
    window_size = window_half_size * 2 + 1
    X, Y = np.meshgrid(np.arange(-window_half_size, window_half_size+1),
                       np.arange(-window_half_size, window_half_size+1))
    window_weight = np.exp(-gaussian_const * (X**2 + Y**2))

    im1H, im1W, _ = im1.shape
    im2H, im2W, _ = im2.shape

    x1_test = x1
    y1_test = y1

    pt = np.array([x1_test, y1_test, 1])
    epiLn2 = F.dot(pt)

    # # create test points
    ptLnDist = np.abs(np.dot(epiLn2, pt)) / \
        np.sqrt(epiLn2[0] ** 2 + epiLn2[1] ** 2)

    sampleDist = np.sqrt((test_radius * 2) ** 2 - ptLnDist ** 2)
    slope = (epiLn2[0] + 1e-100) / (epiLn2[1] + 1e-8)
    closest_x = (epiLn2[1] * (epiLn2[1] * x1_test - epiLn2[0] * y1_test) -
                 epiLn2[0] * epiLn2[2]) / (epiLn2[0] ** 2 + epiLn2[1] ** 2)
    closest_y = -(epiLn2[2] + epiLn2[0] * closest_x)/epiLn2[1]
    x_span = sampleDist / np.sqrt(1 + slope ** 2)
    y_span = sampleDist * slope / np.sqrt(1 + slope ** 2)

    # basically get all points on the epipolar line
    # at a radius of test_radius around (x1, y1)

    # x2_tests = np.linspace(-x_span//2, x_span//2, num=test_radius * 2) + closest_x
    # y2_tests = np.round(-(epiLn2[2] + epiLn2[0] * x2_tests)/epiLn2[1]).astype(int)
    # x2_tests = np.round(x2_tests).astype(int)

    y2_tests = np.linspace(-y_span//2, y_span//2,
                           num=test_radius * 2) + closest_y
    x2_tests = np.round(-(epiLn2[1] * y2_tests +
                        epiLn2[2])/epiLn2[0]).astype(int)
    y2_tests = np.round(y2_tests).astype(int)
    # print([x2_tests, y2_tests])

    # this code previously failed
    # probably because
    # - i forgot the //2
    # - i wrote 1/sqrt(1+sample_dist**2) instead of 1/sqrt(1+slope**2)
    # - i set the radius too small or something

    # remember the principle of KISS
    # y2_tests = np.arange(y1 - test_radius, y1 + test_radius)
    # x2_tests = np.round(-(epiLn2[1]*y2_tests+epiLn2[2])/epiLn2[0]).astype(int)
    # y2_tests = y2_tests.astype(int)

    sel = (x2_tests - window_half_size >= 0) & \
        (x2_tests + window_half_size < im2W) & \
        (y2_tests - window_half_size >= 0) & \
        (y2_tests + window_half_size < im2H)

    x2_tests = x2_tests[sel]
    y2_tests = y2_tests[sel]
    # print([x2_tests, y2_tests])

    # extract the windows and find most similar
    window1 = im1[
        y1_test-window_half_size:y1_test+window_half_size+1,
        x1_test-window_half_size:x1_test+window_half_size+1
    ]
    min_err = np.Inf
    min_x2 = -1
    min_y2 = -1
    for x2_test, y2_test in zip(x2_tests, y2_tests):
        window2 = im2[
            y2_test-window_half_size:y2_test+window_half_size+1,
            x2_test-window_half_size:x2_test+window_half_size+1
        ]

        # calculate similarity, we use euclidean distance
        err = np.sum(np.sum((window1 - window2) ** 2, axis=2) * window_weight)
        if (err < min_err):
            min_err = err
            min_x2 = x2_test
            min_y2 = y2_test

    # Replace pass by your implementation
    return min_x2, min_y2


'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''


def _eval_sevenpoint(args):
    spts1, spts2, M = args
    F = sevenpoint(spts1, spts2, M)
    return F

def ransacF(pts1, pts2, M):
    numPts = pts1.shape[0]
    hpts1 = np.hstack([pts1, np.ones((numPts, 1))])
    hpts2 = np.hstack([pts2, np.ones((numPts, 1))])

    iters = 50
    tol = 2e-3
    best_num_inliers = 0
    best_inliers = None

    sample_points = []
    for _ in range(iters):
        samples = np.random.randint(0, numPts, size=(7, ))
        spts1, spts2 = pts1[samples, :], pts2[samples, :]
        sample_points.append((spts1, spts2, M))

    p = multiprocessing.Pool(multiprocessing.cpu_count())
    F7s = p.map(_eval_sevenpoint, sample_points)
    # F7s = map(_eval_sevenpoint, sample_points)

    for F in F7s:
        for j in range(len(F)):

            # # this error calculates x2.T @ F @ x1 and checks if its 0
            # # doesn't work well
            errs = np.abs(np.sum(hpts2 * (hpts1 @ F[j].T), axis=1))

            # this checks distance from epipolar line
            # epiLn = (hpts1 @ F[j].T)
            # errs = np.abs(np.sum(hpts2 * epiLn, axis=1) / np.sqrt(epiLn[:, 0] ** 2 + epiLn[:, 1] ** 2))

            num_inliers = np.count_nonzero(errs < tol)
            if num_inliers > best_num_inliers:
                print(num_inliers)
                best_num_inliers = num_inliers
                best_inliers = errs < tol

    best_F = eightpoint(pts1[best_inliers, :], pts2[best_inliers, :], M)

    return best_F, best_inliers


'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''


def rodrigues(r):
    norm = np.linalg.norm(r)
    if norm == 0:
        R = np.identity(3)
    elif norm == np.Inf:  # we assume this case doesn't happen
        R = -np.identity(3)
    else:
        rhat = r.reshape((-1,)) / norm
        rhat_cross = np.cross(rhat, np.identity(3) * -1)
        angle = 2 * np.arctan(norm)
        R = np.identity(3) + rhat_cross * np.sin(angle) + \
            (rhat_cross @ rhat_cross) * (1 - np.cos(angle))
    return R


'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''


def invRodrigues(R):
    # assert 1 - 1e-5 < np.abs(np.linalg.det(R)) < 1 + \
    #     1e-5, "det(R) is not 1 or -1"
    # assert np.all(np.abs(R.T @ R - np.identity(3))
    #               < 1e-5), "R is not othrogonal"
    # assert np.all(np.sum(R ** 2, axis=0) - 1 <
    #               1e-5), "Vectors in R are not orthonormal"
    A = (R - R.T)/2
    rho = np.array([A[2, 1], A[0, 2], A[1, 0]])  # same method as wiki
    s = np.linalg.norm(rho)
    c = (np.trace(R) - 1)/2
    if c == 1:
        r = np.array([0, 0, 0])
    elif c == -1:
        r = np.array([np.Inf, np.Inf, np.Inf])  # :D
    else:
        rhat = rho / s
        angle = np.arctan2(s, c)
        r = rhat * np.tan(angle/2)
    return r


'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''


def rodriguesResidual(K1, M1, p1, K2, p2, x):

    assert p1.shape[0] == p2.shape[0]
    num_pts = p1.shape[0]
    assert x.shape[0] == num_pts * 3 + 6

    P = x[:3*num_pts].reshape((-1, 3))
    r2 = x[3*num_pts:3*num_pts+3]
    t2 = x[3*num_pts+3:]
    M2 = np.hstack([rodrigues(r2), t2[:, np.newaxis]])

    threeD_homo = np.hstack([P, np.ones((num_pts, 1))]).T
    p1_hat = ((K1 @ M1) @ threeD_homo).T
    p1_hat = p1_hat[:, :2] / (p1_hat[:, [2]] + 1e-100)
    p2_hat = ((K2 @ M2) @ threeD_homo).T
    p2_hat = p2_hat[:, :2] / (p2_hat[:, [2]] + 1e-100)
    residuals = np.concatenate(
        [(p1-p1_hat).reshape([-1]), (p2-p2_hat).reshape([-1])]).reshape((-1, 1))
    return residuals


'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''


def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):

    def residual(x):
        return rodriguesResidual(K1, M1, p1, K2, p2, x).reshape((-1, ))

    num_pts = p1.shape[0]

    r2_init = invRodrigues(M2_init[:3, :3])
    t2_init = M2_init[:, 3]
    x_init = np.hstack([P_init.reshape((-1,)), r2_init, t2_init])

    x_opt, _ = scipy.optimize.leastsq(residual, x_init)
    P_opt = x_opt[:3*num_pts].reshape((-1, 3))
    r2_opt = x_opt[3*num_pts:3*num_pts+3]
    t2_opt = x_opt[3*num_pts+3:]
    M2_opt = np.hstack([rodrigues(r2_opt), t2_opt[:, np.newaxis]])

    print("init_err:", np.linalg.norm(rodriguesResidual(K1, M1, p1, K2, p2, x_init))**2)
    print("opt_err:", np.linalg.norm(rodriguesResidual(K1, M1, p1, K2, p2, x_opt))**2)
    return M2_opt, P_opt


if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    import submission as sub
    import helper
    import findM2
    data = np.load('../data/some_corresp.npz')
    data_noisy = np.load('../data/some_corresp_noisy.npz')
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')
    with np.load('../data/intrinsics.npz') as intrinsics:
        K1 = intrinsics['K1']
        K2 = intrinsics['K2']

    N = data['pts1'].shape[0]
    N_noisy = data_noisy['pts1'].shape[0]
    M = 640

    run_question = input("""Which question do you want to run?
"2.1" - The Eight Point Algorithm
"2.2" - The Seven Point Algorithm
"3.1" - Essential Matrix
"4.1" - Epipolar Correspondences
"5.1" - Ransac Fundamental Matrix Estimation
"5.2" - Rodrigues Conversion
"5.3" - Bundle Adjustment
"combined" - Bundle Adjustment with Correspondence Matching (q4.1+q5.3)
""")
    
    # q2.1
    if run_question == "2.1":
        F8 = sub.eightpoint(data['pts1'], data['pts2'], M)
        print(F8)
        np.savez("q2_1.npz", F=F8, M=M)
        helper.displayEpipolarF(im1, im2, F8)
        # # Checking whether this satisfies x2^T @ F @ x1 = 0
        # errs = []
        # for i in range(N):
        #     pt1 = data['pts1'][i, :]
        #     pt2 = data['pts2'][i, :]
        #     err = np.array([[pt2[0], pt2[1], 1]]) @ F8 @ np.array([[pt1[0]], [pt1[1]], [1]])
        #     print(f"err{i} =", err)
        #     errs.append(np.abs(err[0, 0]))
        # print('err sum =', np.sum(errs))

    # q2.2
    elif run_question == "2.2":

        # best sample
        # samples = [50, 55, 25, 22, 61, 47, 70]
        # spts1, spts2 = data['pts1'][samples, :], data['pts2'][samples, :]
        # F7 = sub.sevenpoint(spts1, spts2, M)
        # print("best F7:", F7[0])
        # np.savez("q2_2.npz", F=F7[0], M=M, pts1=spts1, pts2=spts2)
        # helper.displayEpipolarF(im1, im2, F7[0])

        samples = np.random.randint(0, data['pts1'].shape[0]-1, size=(7, ))
        print(list(samples))
        spts1, spts2 = data['pts1'][samples, :], data['pts2'][samples, :]
        F7 = sub.sevenpoint(spts1, spts2, M)
        print(F7)
        helper.displayEpipolarF(im1, im2, F7[0])
        # # test error of seven point algo
        # errs = []
        # for i in range(7):
        #     for F in F7:
        #         err = np.array([[spts2[i, 0], spts2[i, 1], 1]]) @ F @ np.array([[spts1[i, 0]], [spts1[i, 1]], [1]])
        #         print(f"err{i} =", err)
        #         errs.append(np.abs(err[0, 0]))
        # print('max err =', np.max(errs))

    # q3.1
    elif run_question == "3.1":
        F8 = sub.eightpoint(data['pts1'], data['pts2'], M)
        E = essentialMatrix(F8, K1, K2)
        print(E)

    # q4.1
    elif run_question == "4.1":
        F8 = sub.eightpoint(data['pts1'], data['pts2'], M)
        helper.epipolarMatchGUI(im1, im2, F8)

    # q5.1
    elif run_question == "5.1":
        # F8 = sub.eightpoint(data_noisy['pts1'], data_noisy['pts2'], M)
        # print(F8)
        # helper.displayEpipolarF(im1, im2, F8)

        F, inliers = sub.ransacF(data_noisy['pts1'], data_noisy['pts2'], M)
        print(list(F), list(inliers))
        helper.displayEpipolarF(im1, im2, F)

    # q5.2
    elif run_question == "5.2":
        for _ in range(1000):
            r = np.random.randn(3)
            # r = np.array([0, 0, 0])
            R = rodrigues(r)
            # print(R, np.linalg.det(R))
            r2 = invRodrigues(rodrigues(r))
            print(r, r2)
            if not np.all(np.abs(r - r2) < 1e-12):
                input()
        
    # q5.3
    elif run_question == "5.3":
        
        data = data_noisy
        N = N_noisy

        # ransac result
        inliers = [True, True, True, True, True, True, False, True, True, 
                    True, True, True, True, True, True, True, True, False, 
                    False, True, True, True, True, False, True, False, True,
                    True, True, False, False, True, True, True, False, False, 
                    True, True, False, True, False, True, False, False, True,
                    False, True, True, True, False, False, True, True, True,
                    True, True, True, False, True, True, True, True, True,
                    True, True, True, True, True, False, True, False, False,
                    True, False, False, False, True, True, True, True, True,
                    False, True, False, True, False, False, True, True, False,
                    True, True, True, True, True, True, True, True, True, True,
                    False, True, False, False, True, False, False, True, True,
                    True, True, False, False, True, True, True, True, True, True,
                    True, True, True, True, True, False, True, False, True, True,
                    True, True, True, True, False, False, True, False, True, False, True]
        inliers_p1 = data['pts1'][inliers]; inliers_p2 = data['pts2'][inliers]
        F = eightpoint(inliers_p1, inliers_p2, M)

        # or, run ransac directly
        # F, inliers = sub.ransacF(data['pts1'], data['pts2'], M)
        # inliers_p1 = data['pts1'][inliers]
        # inliers_p2 = data['pts2'][inliers]

        M1 = np.hstack([np.eye(3), np.zeros((3, 1))])
        M2_init, P_init = findM2.findM2(F, K1, M1, inliers_p1, K2, inliers_p2)
        M2_opt, P_opt = bundleAdjustment(K1, M1, inliers_p1, K2, M2_init, inliers_p2, P_init)
        _, err_init = triangulate(K1@M1, inliers_p1, K2@M2_init, inliers_p2)
        _, err_opt = triangulate(K1@M1, inliers_p1, K2@M2_opt, inliers_p2)
        print("Initial projection error:", err_init)
        print("Projection error after optimization:", err_opt)
        findM2.plotP(P_init)
        findM2.plotP(P_opt)
        R2_opt = M2_opt[:3, :3]
        t2_opt = M2_opt[:, 3]
        E_opt = R2_opt @ np.array([[0, -t2_opt[2], t2_opt[1]],
                                   [t2_opt[2], 0, t2_opt[0]],
                                   [-t2_opt[1], t2_opt[0], 0]])
        F_opt = np.linalg.inv(K2.T) @ E_opt @ np.linalg.inv(K1)
        helper.displayEpipolarF(im1, im2, F_opt)

    elif run_question == "combined":
    
        # extra: temple coords
        F = np.array([[-8.331492341800977e-09, 1.2953846201515214e-07, -0.0011718785098119202], 
                    [6.513583362032381e-08, 5.706700587341282e-09, -4.134350366704326e-05], 
                    [0.0011307876458092711, 1.918236366032668e-05, 0.004168620793724458]])
        M1 = np.hstack([np.eye(3), np.zeros((3, 1))])
        M2 = np.array([[0.9993695037930264, 0.035197572534239496, -0.004661091736671946, 0.01780295850034071], 
                    [-0.03283857784245489, 0.9662338863618545, 0.255565460599339, -1.0], 
                    [0.013498988620105281, -0.25525126392197206, 0.9667805177869844, 0.08705061683747736]])

        temple_coords = np.load('../data/templeCoords.npz')
        x1 = temple_coords['x1']; y1 = temple_coords['y1']
        num_pts = x1.shape[0]
        pts1 = np.hstack([x1, y1])
        pts2 = np.vstack([sub.epipolarCorrespondence(im1, im2, F, x1[i, 0], y1[i, 0]) for i in range(num_pts)])
        P, err = sub.triangulate(K1 @ M1, pts1, K2 @ M2, pts2)
        M2_opt, P_opt = bundleAdjustment(K1, M1, pts1, K2, M2, pts2, P)
        findM2.plotP(P)
        findM2.plotP(P_opt)


