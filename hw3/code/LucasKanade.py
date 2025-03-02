import numpy as np
from scipy.ndimage import shift, sobel
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	p0: Initial movement vector [dp_x0, dp_y0]
	# Output:
	#	p: movement vector [dp_x, dp_y]

	# Put your implementation here
	p = p0.copy()
	x1, y1, x2, y2 = rect

	# get integer pixel points
	rectH, rectW = int(np.around(y2-y1+1)), int(np.around(x2-x1+1))
	pixelsX, pixelsY = np.linspace(x1, x2, rectW), np.linspace(y1, y2, rectH)
	meshX, meshY = np.meshgrid(pixelsX, pixelsY)

	# interpolate the images
	imH, imW = It.shape
	It_intrep = RectBivariateSpline(np.arange(imH), np.arange(imW), It) # ! check
	It1_intrep = RectBivariateSpline(np.arange(imH), np.arange(imW), It1) # ! check

	for _ in range(20):
		# Construct A and b
		# It_rect = It[y1:y2+1, x1:x2+1]
		# It1_rect = shift(It1, -p)[y1:y2+1, x1:x2+1]
		# It1_grad_x = shift(sobel(It1, axis=1), -p)[y1:y2+1, x1:x2+1]
		# It1_grad_y = shift(sobel(It1, axis=0), -p)[y1:y2+1, x1:x2+1]
		It_rect = It_intrep.ev(meshY, meshX)
		It1_rect = It1_intrep.ev(meshY + p[1], meshX + p[0])
		It1_grad_x = It1_intrep.ev(meshY + p[1], meshX + p[0], dy=1) # we assume scipy's x means the first
		It1_grad_y = It1_intrep.ev(meshY + p[1], meshX + p[0], dx=1) # axis, i.e. y-axis in the image
		b = np.reshape(It_rect - It1_rect, -1)
		A = np.vstack([It1_grad_x.reshape(-1), It1_grad_y.reshape(-1)]).T

		# Evaluate the least-squares solution of AΔp=b
		p_step = np.linalg.lstsq(A, b)[0]

		# Update our offset by p←p+Δp
		if np.linalg.norm(p_step) > 1e-10: # 0.7 works well, 0.1 makes no visible difference
			p += p_step	
		else:
			# print(f"Number of iterations: {_}")
			break
	return p	
