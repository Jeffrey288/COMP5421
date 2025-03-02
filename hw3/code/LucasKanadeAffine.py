import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image
	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]
    # put your implementation here

	p = np.zeros(6)

	# interpolate the images
	imH, imW = It.shape
	orgY, orgX = np.meshgrid(np.arange(imH), np.arange(imW))
	It_intrep = RectBivariateSpline(np.arange(imH), np.arange(imW), It) # ! check
	It1_intrep = RectBivariateSpline(np.arange(imH), np.arange(imW), It1) # ! check

	for _ in range(20):
		affineX = (p[0]+1) * orgX + p[1] * orgY + p[2]
		affineY = p[3] * orgX + (p[4]+1) * orgY + p[5]
		select = ((0 <= affineX) & (affineX < imW)) & ((0 <= affineY) & (affineY < imH))
		orgX_selected, orgY_selected = orgX[select], orgY[select]
		affineX_selected, affineY_selected = affineX[select], affineY[select]

		# Construct A and b
		It_rect = It_intrep.ev(orgY_selected, orgX_selected)
		It1_rect = It1_intrep.ev(affineY_selected, affineX_selected)
		It1_grad_x = It1_intrep.ev(affineY_selected, affineX_selected, dy=1) # we assume scipy's x means the first
		It1_grad_y = It1_intrep.ev(affineY_selected, affineX_selected, dx=1) # axis, i.e. y-axis in the image
		b = It_rect - It1_rect
		A = np.vstack([It1_grad_x * orgX_selected, It1_grad_x * orgY_selected, It1_grad_x, 
		 			   It1_grad_y * orgX_selected, It1_grad_y * orgY_selected, It1_grad_y]).T

		# Evaluate the least-squares solution of AΔp=b
		p_step = np.linalg.lstsq(A, b, rcond=None)[0]

		# Update our offset by p←p+Δp
		if np.linalg.norm(p_step) > 1e-4: 
			p += p_step	
		else:
			print(f"Number of iterations: {_}")
			break

	# affine_matrix = [[p[4]+1, p[3], p[5]],
	# 				 [p[1], p[0]+1, p[2]]] # for scipy
	affine_matrix = [[p[0]+1, p[1], p[2]],
					 [p[3], p[4]+1, p[5]]] # our conventions
	return np.array(affine_matrix)
