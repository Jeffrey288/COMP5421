import numpy as np
import scipy.ndimage
import os,time
import skimage.transform

def extract_deep_feature(x,vgg16_weights):
	'''
	Extracts deep features from the given VGG-16 weights.

	[input]
	* x: numpy.ndarray of shape (H,W,3)
	* vgg16_weights: numpy.ndarray of shape (L,3)

	[output]
	* feat: numpy.ndarray of shape (K)
	'''
	# i am paranoid
	# fix image channel and data type inconsistencies
	image = x
	if len(image.shape) < 3:
		image = image[:, :, np.newaxis]
	if image.shape[2] != 3:
		if image.shape[2] > 3:
			image = image[:, :, :3]
		elif image.shape[2] == 1:
			image = np.repeat(image, 3, axis=2)
		print("image does not have 3 channels")
	if not isinstance(image[0, 0, 0].item(), float):
		image = image.astype(float)
		print("image does not have float datatype")
	if np.max(image) > 1:
		image = image / 255
		print("image data range is not from 0 to 1")
	x = image

	# VGG-16 assumes that all input imagery to the network 
	# is resized to 224×224 with the three color channels preserved
	# (use skimage.transform.resize() to do this before passing any 
	# imagery to the network). 
	resized_x = skimage.transform.resize(x, (224, 224))

	# And be sure to normalize the image using suggested 
	# mean and std before extracting the feature:
	mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis, :]
	std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis, :]
	normalized_x = (resized_x - mean) / std

	# loop through the layers
	res = normalized_x
	linear_layers = 0
	for layer in vgg16_weights:
		if layer[0] == "conv2d":
			res = multichannel_conv2d(res, layer[1], layer[2])
		elif layer[0] == "maxpool2d":
			res = max_pool2d(res, layer[1])
		elif layer[0] == "relu":
			res = relu(res)
		elif layer[0] == "linear":
			if len(res.shape) != 1: res = res.reshape((-1, ))
			res = linear(res, layer[1], layer[2])
			linear_layers += 1
			if linear_layers == 2: break
		print(layer[0], res.shape)

	return res # shape (4096,)

def multichannel_conv2d(x,weight,bias):
	'''
	Performs multi-channel 2D convolution.

	[input]
	* x: numpy.ndarray of shape (H,W,input_dim)
	* weight: numpy.ndarray of shape (output_dim,input_dim,kernel_size,kernel_size)
	* bias: numpy.ndarray of shape (output_dim)

	[output]
	* feat: numpy.ndarray of shape (H,W,output_dim)
	'''
	image_dim = x.shape[:2]
	input_dim = x.shape[2]
	output_dim = weight.shape[0]
	feat = np.zeros((*image_dim, output_dim))
	for j in range(output_dim): # is there a better option than to for loop over these...
		for k in range(input_dim):
			feat[:, :, j] += scipy.ndimage.convolve(x[:, :, k], weight[j, k])
		feat[:, :, j] += bias[j]
	return feat

def relu(x):
	'''
	Rectified linear unit.

	[input]
	* x: numpy.ndarray

	[output]
	* y: numpy.ndarray
	'''
	return np.maximum(x, 0)

def max_pool2d(x,size):
	'''
	2D max pooling operation.

	[input]
	* x: numpy.ndarray of shape (H,W,input_dim)
	* size: pooling receptive field

	[output]
	* y: numpy.ndarray of shape (H/size,W/size,input_dim)
	'''
	# reference: https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy
	newH = x.shape[0] // size
	newW = x.shape[1] // size
	return x[:size*newH, :size*newW, :].reshape((newH, size, newW, size, -1)).max(axis=(1, 3))

def linear(x,W,b):
	'''
	Fully-connected layer.

	[input]
	* x: numpy.ndarray of shape (input_dim)
	* weight: numpy.ndarray of shape (output_dim,input_dim)
	* bias: numpy.ndarray of shape (output_dim)

	[output]
	* y: numpy.ndarray of shape (output_dim)
	'''
	return W @ x + b
