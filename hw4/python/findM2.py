'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
import matplotlib.pyplot as plt
import submission as sub
import helper

def findM2(F, K1, M1, p1, K2, p2):

    E = sub.essentialMatrix(F, K1, K2)
    # Checking if the essential matrix is correctly computed
    # assert np.all(np.abs(F8 - np.linalg.inv(K2.T) @ E @ np.linalg.inv(K1)) < 1e-5), \
    #     f"discrepancy: F={F8} vs {np.linalg.inv(K2.T) @ E @ np.linalg.inv(K1)}"

    M2s = helper.camera2(E)
    Ps = []
    count = []
    for i in range(4):
        # get the Ms through [:, :, n]
        M2 = M2s[:, :, i]

        C1 = K1 @ M1
        C2 = K2 @ M2

        P, err = sub.triangulate(C1, p1, C2, p2)
        print(err)

        # The method to disambiguate camera pose is NOT minimizing
        # reproject error: they are all similar and only reflect
        # the validity of homogenous equation Ah = 0

        # https://inst.eecs.berkeley.edu/~ee290t/fa19/lectures/lecture10-3-decomposing-F-matrix-into-Rotation-and-Translation.pdf
        cond = (P + M2[:, [3]].T @ M2[:, :3]) @ M2[:, [2]]
        this_count = np.count_nonzero(cond > 0)

        Ps.append(P)
        count.append(this_count)

    min_ind = count.index(max(count))
    print(min_ind)
    P = Ps[min_ind]
    M2 = M2s[:, :, min_ind]
    return M2, P

def plotP(P):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], marker="o")
    plt.show()
    
if __name__ == "__main__":
    data = np.load('../data/some_corresp.npz')
    with np.load('../data/intrinsics.npz') as intrinsics:
        K1 = intrinsics['K1']
        K2 = intrinsics['K2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    N = data['pts1'].shape[0]
    M = 640

    F8 = sub.eightpoint(data['pts1'], data['pts2'], M)
    
    # print(F8.tolist())
    # Checking whether this satisfies x2^T @ F @ x1 = 0
    # errs = []
    # for i in range(N):
    #     pt1 = data['pts1'][i, :]
    #     pt2 = data['pts2'][i, :]
    #     err = np.array([[pt2[0], pt2[1], 1]]) @ F8 @ np.array([[pt1[0]], [pt1[1]], [1]])
    #     print(f"err{i} =", err)
    #     errs.append(np.abs(err[0, 0]))
    # print('max err =', np.max(errs))
    # input()

    M1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    M2, P = findM2(F8, K1, M1, data['pts1'], K2, data['pts2'])

    plotP(P)

    np.savez('q3_3.npz',
            P=P,
            C2=K2@M2,
            M2=M2)
