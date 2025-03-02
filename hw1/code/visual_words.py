import numpy as np
import multiprocessing
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os,time
import matplotlib.pyplot as plt
import util
import random
import numpy.matlib as matlib

alpha = 200
K = 150

def extract_filter_responses(image):
	'''
	Extracts the filter responses for the given image.

	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)
	[output]
	* filter_responses: numpy.ndarray of shape (H,W,3F)
	'''
	# fix image channel and data type inconsistencies
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
	
	# lab
	image = skimage.color.rgb2lab(image)
	
	# apply the filters
	scales = [1, 2, 4, 8, 8*np.sqrt(2)]
	gaussian = [scipy.ndimage.gaussian_filter(image, (scale, scale, 0)) for scale in scales]
	gaussian_laplace = [np.dstack([scipy.ndimage.gaussian_laplace(image[:,:,i], scale) for i in range(3)]) for scale in scales]
	derv_gaussian_x = [scipy.ndimage.sobel(gimg, axis=1) for gimg in gaussian]
	derv_gaussian_y = [scipy.ndimage.sobel(gimg, axis=0) for gimg in gaussian]
	filter_responses = []
	for i in range(len(scales)):
		filter_responses.append(gaussian[i])
		filter_responses.append(gaussian_laplace[i])
		filter_responses.append(derv_gaussian_x[i])
		filter_responses.append(derv_gaussian_y[i])
	filter_responses = np.dstack(filter_responses)
	# print(filter_responses.shape)
	
	return filter_responses

def get_visual_words(image,dictionary):
	'''
	Compute visual words mapping for the given image using the dictionary of visual words.

	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)
	
	[output]
	* wordmap: numpy.ndarray of shape (H,W)
	'''
	
	filter_responses = extract_filter_responses(image)

	# cdist approach, idk why but very freaking fast
	filter_responses = filter_responses.reshape((-1, filter_responses.shape[-1]))
	dist = scipy.spatial.distance.cdist(filter_responses, dictionary, 'euclidean')
	wordmap = np.take(np.arange(dictionary.shape[0]), np.argsort(dist, axis=1)[:, 0])
	wordmap = wordmap.reshape(image.shape[:2])

	# Alternative solution
	# temp_dict = np.expand_dims(dictionary, axis=0)
	# wordmap = np.zeros(image.shape[:2])
	# for r in range(image.shape[0]):
	# 	print(r)
	# 	temp_response = np.expand_dims(filter_responses[r], axis=1)
	# 	dist = np.power(np.sum(np.power(temp_response - temp_dict, 2), axis=2), 1/2)
	# 	wordmap[r, :] = np.take(np.arange(dictionary.shape[0]), np.argsort(dist, axis=1)[:, 0])

	return wordmap

def compute_dictionary_one_image(args):
	'''
	Extracts random samples of the dictionary entries from an image.
	This is a function run by a subprocess.

	[input]
	* i: index of training image
	* alpha: number of random samples
	* image_path: path of image file
	* time_start: time stamp of start time

	[saved]
	* sampled_response: numpy.ndarray of shape (alpha,3F)
	'''

	i, alpha, image_path = args
	print(f"Response generating for the {i}-th image.")
	# print(i, alpha, image_path)

	image = skimage.io.imread(f"../data/{image_path}")
	image = image.astype('float')/255
	filter_responses = extract_filter_responses(image)

	imH, imW = image.shape[0], image.shape[1]
	sampleH = np.random.randint(0, imH-1, size=(alpha, ))
	sampleW = np.random.randint(0, imW-1, size=(alpha, ))
	sampled_responses = filter_responses[sampleH, sampleW, :]

	np.save(f'../temp/visual_words/{i}.npy', sampled_responses)


def compute_dictionary(num_workers=2):
	'''
	Creates the dictionary of visual words by clustering using k-means.

	[input]
	* num_workers: number of workers to process in parallel
	
	[saved]
	* dictionary: numpy.ndarray of shape (K,3F)
	'''

	train_data = np.load("../data/train_data.npz", allow_pickle=True)

	labels = train_data['labels']
	image_names = train_data['image_names']
	num_train_data = image_names.shape[0]

	force_run = True
	if force_run:
		run_indices = list(range(num_train_data))
	else:
		run_indices = [i for i in range(num_train_data) if not os.path.exists(f'../temp/visual_words/{i}.npy')]
	run_args = [(i, alpha, image_names[i][0]) for i in run_indices]
	

	if run_args:
		print(f"Generation starting, total {len(run_indices)} images to be processed...")
		
		# without multiprocessing
		# for args in run_args:
		# 	compute_dictionary_one_image(args)

		# with multiprocessing
		p = multiprocessing.Pool(num_workers)
		with p:
			p.map(compute_dictionary_one_image, run_args)
	else:
		print("All responses generated already.")

	
	filter_responses = [np.load(f'../temp/visual_words/{i}.npy') for i in range(num_train_data)]
	filter_responses = np.vstack(filter_responses)
	print("Responses loaded.")

	kmeans = sklearn.cluster.KMeans(n_clusters=K, verbose=1).fit(filter_responses)
	dictionary = kmeans.cluster_centers_
	
	np.save('dictionary.npy', dictionary)
	print("Dictionary saved to dictionary.npy.")

	return dictionary

