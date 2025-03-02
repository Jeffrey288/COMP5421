import numpy as np
from planarH import computeH

def compute_extrinsics(K, H):
    """
    @params
    K - intrinsic camera parameters, 3X3 matrix
    H - estimate homography

    row for z is missing because it is 0
    the homography matrix H = K[R[:, :2], t]
    """
    # Assuming the lines on H lie on a single point
    Rt = np.linalg.inv(K) @ H
    U, S, Vh = np.linalg.svd(Rt[:, :2])
    R = np.zeros((3, 3))
    R[:, :2] = U @ np.array([[1,0],[0,1],[0,0]]) @ Vh
    R[:, 2] = np.cross(R[:, 0], R[:, 1])
    if (np.linalg.det(R) < 0): 
        R[:, 2] *= -1
    scaling_factor = np.average(Rt[:, :2] / R[:, :2])
    t = Rt[:, 2] / scaling_factor
    return R, t

def project_extrinsics(K, W, R, t):
    X = K @ np.hstack([R, t[:, np.newaxis]]) @ np.vstack([W, np.ones((1, W.shape[1]))])
    # print(X[2, :])
    return (X[:2, :].T / X[[2], :].T).T

if __name__ == "__main__":
    K = np.array([
            [3043.72, 0.0, 1196.00],
            [0.0, 3043.72, 1604.00],
            [0.0, 0.0, 1.0]
        ])
    X = np.array([
            [483, 1704, 2175, 67],
            [810, 781, 2217, 2286]
        ])
    W = np.array([  
            [0.0, 18.2, 18.2, 0.0], 
            [0.0, 0.0, 26.0, 26.0],
            [0.0, 0.0, 0.0, 0.0]
        ])

    H = computeH(X, W[:2, :])
    R, t = compute_extrinsics(K, H)
    print(R, t)

    f = open("../data/sphere.txt")
    data = f.read().strip()
    sphere_world_coords = np.array(list(map(float, data.split()))).reshape((3, -1))
    sphere_center_coords = np.average(sphere_world_coords, axis=1)[:, np.newaxis]
    print("center", sphere_center_coords)

    # lambda * O = K @ [R[:, :2], t] @ W
    O_im_coords = np.array([[817], [1619], [1]])
    O_world_coords_xy = np.linalg.inv(K @ np.hstack([R[:, :2], t[:, np.newaxis]])) @ O_im_coords
    print(O_world_coords_xy)
    O_world_coords = np.vstack([O_world_coords_xy[:2, :] / O_world_coords_xy[2, 0], [6.858/2]])
    # print(O_world_coords)
    
    sphere_translated_world_coords = sphere_world_coords - sphere_center_coords + O_world_coords
    # print(np.max(sphere_translated_world_coords, axis=1))
    # print(np.max(sphere_world_coords, axis=1))
    # X = project_extrinsics(K, sphere_world_coords, R, t)
    X = project_extrinsics(K, sphere_translated_world_coords, R, t)

    import cv2
    import matplotlib.pyplot as plt
    im = cv2.imread('../data/prince_book.jpeg')
    plt.imshow(im)
    plt.plot(X[0, :], X[1, :], 'ro', markersize=1)
    plt.show()

