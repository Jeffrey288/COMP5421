import numpy as np
import scipy.ndimage
import skimage.filters
import LucasKanadeAffine
import InverseCompositionAffine


def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1 
	# Output:
	#	mask: [nxm]
	# put your implementation here

	# -=-=- Uncomment the version you want to run -=-=-=-
	# M = InverseCompositionAffine.InverseCompositionAffine(image1, image2)
	M = LucasKanadeAffine.LucasKanadeAffine(image1, image2)
	# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

	M_scipy = [[M[1, 1], M[1, 0], M[1, 2]],
    	       [M[0, 1], M[0, 0], M[0, 2]],
			   [0.0, 0.0, 1.0]]
	M_scipy_inv = np.linalg.inv(M_scipy)[:2]
	transformed_image1 = scipy.ndimage.affine_transform(image1, matrix=M_scipy_inv, mode='constant', cval=np.inf)
	isInf = transformed_image1 == np.inf 		# patch the areas outside the boundary of the image
	transformed_image1[isInf] = image2[isInf] 	# with the second image
	transformed_image1[237:] = image2[237:]	# some lower boundary stuff
	# i don't like hardcoding stuff but sorry this is a pain to look at

	# Evaluate Lucas-Kanade correctness
	# import matplotlib.pyplot as plt
	# fig, ax = plt.subplots(2, 3)
	# ax[0, 0].imshow(image1)
	# ax[0, 1].imshow(image2)
	# ax[1, 0].imshow(transformed_image1, interpolation=None)
	# ax[1, 0].set_title('Transformed preceeding frame')
	# ax[1, 1].imshow(np.abs(transformed_image1 - image2))
	# ax[1, 1].set_title('Residual between transformed preceeding frame and current frame')
	# ax[1, 2].imshow(np.abs(image1 - image2))
	# ax[1, 2].set_title('Residual without transformation')
	# plt.show()

	diff = np.abs(transformed_image1 - image2)
	# diff = scipy.ndimage.gaussian_filter(diff, sigma=0.05)
	mask = diff > 0.1
	mask = scipy.ndimage.binary_erosion(mask, np.ones((2, 2)))
	mask = scipy.ndimage.binary_dilation(mask, scipy.ndimage.generate_binary_structure(2, 8), iterations=5)
	# View the mask
	# fig, ax = plt.subplots(2, 2)
	# ax[0, 0].imshow(np.abs(transformed_image1 - image2))
	# ax[1, 0].imshow(diff)
	# ax[0, 1].imshow(mask, cmap="gray")
	# plt.show()

	return mask


