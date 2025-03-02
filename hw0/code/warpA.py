import numpy as np

def warp(im, A, output_shape):
    """ Warps (h,w) image im using affine (3,3) matrix A
    producing (output_shape[0], output_shape[1]) output image
    with warped = A*input, where warped spans 1...output_size.
    Uses nearest neighbor interpolation."""

    # import cv2 as cv
    # new_im = cv.warpAffine(im, A[:2, :], dsize=output_shape[::-1])

    # loop-free implementation (treating rows as y coordinate) -----------------

    # x = np.arange(output_shape[1])
    # y = np.arange(output_shape[0])
    # mesh_d = np.array(np.meshgrid(x, y) + [np.ones(output_shape)], dtype=np.int16).transpose([1, 2, 0])
    # mesh_d.shape = mesh_d.shape + (1, )
    # mesh_s = np.matmul(np.linalg.inv(A), mesh_d)
    # mesh_s = np.round(mesh_s[:,:]).astype(int)

    # # flat_source = 
    # flat_dim = (output_shape[0] * output_shape[1], 3)
    # flat_dest = mesh_d.reshape(flat_dim, order='F')
    # flat_source = mesh_s.reshape(flat_dim, order='F')
    # isValid = np.logical_and(
    #     np.logical_and(flat_source[:, 0] >= 0, flat_source[:, 0] < im.shape[1]),
    #     np.logical_and(flat_source[:, 1] >= 0, flat_source[:, 1] < im.shape[0])
    # )
    # flat_dest = flat_dest[isValid]
    # flat_source = flat_source[isValid]

    # new_im = np.zeros(output_shape)
    # new_im[flat_dest[:, 1], flat_dest[:, 0]] = im[flat_source[:, 1], flat_source[:, 0]]

    # loop-free implementation (treating rows as x coordinate) -----------------

    x = np.arange(output_shape[0])
    y = np.arange(output_shape[1])
    mesh_d = np.array(np.meshgrid(x, y) + [np.ones(output_shape[::-1])], dtype=np.int16).transpose([1, 2, 0])
    mesh_d.shape = mesh_d.shape + (1, )
    mesh_s = np.matmul(np.linalg.inv(A), mesh_d)
    mesh_s = np.round(mesh_s[:,:]).astype(int)

    # flat_source = 
    flat_dim = (output_shape[0] * output_shape[1], 3)
    flat_dest = mesh_d.reshape(flat_dim, order='F')
    flat_source = mesh_s.reshape(flat_dim, order='F')
    isValid = np.logical_and(
        np.logical_and(flat_source[:, 0] >= 0, flat_source[:, 0] < im.shape[0]), # x = rows
        np.logical_and(flat_source[:, 1] >= 0, flat_source[:, 1] < im.shape[1])  # y = cols
    )
    flat_dest = flat_dest[isValid]
    flat_source = flat_source[isValid]

    new_im = np.zeros(output_shape)
    new_im[flat_dest[:, 0], flat_dest[:, 1]] = im[flat_source[:, 0], flat_source[:, 1]]  

    # implementation using for loop -------------------------------

    # new_im = np.zeros(output_shape)
    # for x in range(output_shape[1]):
    #     for y in range(output_shape[0]):
    #         p_d = np.array([x, y, 1]).reshape((3, 1))
    #         p_s = np.round(A @ p_d).astype(int).flatten()
    #         if (0 <= p_s[1] < im.shape[0] and 0 <= p_s[0] < im.shape[1]):
    #             new_im[y, x] = im[p_s[1], p_s[0]]

    return new_im
