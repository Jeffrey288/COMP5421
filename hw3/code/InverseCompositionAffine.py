import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image

	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]

	p = np.zeros(6)
	M_init = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

	# interpolate the images
	imH, imW = It.shape
	margin_scale = 0.9
	marginH, marginW = int((imH * (1- margin_scale)) / 2), int((imW * (1- margin_scale)) / 2) # 0.9 pixels
	orgY, orgX = np.meshgrid(np.arange(marginH, imH-marginH), np.arange(marginW, imW-marginW))
	orgY, orgX = orgY.flatten(), orgX.flatten()
	It_intrep = RectBivariateSpline(np.arange(imH), np.arange(imW), It) # ! check
	It1_intrep = RectBivariateSpline(np.arange(imH), np.arange(imW), It1) # ! check

	# presample the A
	It_rect = It_intrep.ev(orgY, orgX)
	It_grad_x = It_intrep.ev(orgY, orgX, dy=1) # we assume scipy's x means the first
	It_grad_y = It_intrep.ev(orgY, orgX, dx=1) # axis, i.e. y-axis in the image
	A = np.vstack([It_grad_x * orgX, It_grad_x * orgY, It_grad_x, 
				   It_grad_y * orgX, It_grad_y * orgY, It_grad_y]).T
	Ainv = np.linalg.pinv(A)

	for _ in range(20):
		affineX = (p[0]+1) * orgX + p[1] * orgY + p[2]
		affineY = p[3] * orgX + (p[4]+1) * orgY + p[5]

		# Construct b
		It1_rect = It1_intrep.ev(affineY, affineX)
		b = It1_rect - It_rect

		# Evaluate the least-squares solution of AΔp=b
		p_step = Ainv @ b

		# Update our offset by p←p+Δp
		if np.linalg.norm(p_step) > 1e-4: 
			M_step = [[p_step[0]+1, p_step[1], p_step[2]],
					  [p_step[3], p_step[4]+1, p_step[5]],
					  [0.0, 0.0, 1.0]] # our conventions
			M_step_inv = np.linalg.inv(M_step)
			M_step_inv = M_step_inv / M_step_inv[2, 2] # scale it back to 0 0 1
			M_cur = [[p[0]+1, p[1], p[2]],
					 [p[3], p[4]+1, p[5]],
					 [0.0, 0.0, 1.0]]
			M_new = M_cur @ M_step_inv
			p = (M_new[:2] - M_init).reshape(p.shape)
		else:
			print(f"Number of iterations: {_}")
			break

	# affine_matrix = [[p[4]+1, p[3], p[5]],
	# 				 [p[1], p[0]+1, p[2]]] # for scipy
	affine_matrix = [[p[0]+1, p[1], p[2]],
					 [p[3], p[4]+1, p[5]]] # our conventions
	return np.array(affine_matrix)

