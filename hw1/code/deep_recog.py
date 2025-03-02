import numpy as np
import multiprocessing
import threading
import queue
import imageio
import os,time
import torch
import skimage.transform
import torchvision.transforms
import torchvision
import util
import network_layers

K_vgg16 = 4096
def build_recognition_system(vgg16,num_workers=2):
	'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N,K)
	* labels: numpy.ndarray of shape (N)
	'''

	train_data = np.load("../data/train_data.npz", allow_pickle=True)

	labels = train_data['labels']
	image_names = train_data['image_names']
	num_train_data = image_names.shape[0]

	force_run = True
	if force_run:
		run_indices = list(range(num_train_data))
	else:
		run_indices = [i for i in range(num_train_data) if not os.path.exists(f'../temp/deep_features/{i}.npy')]
	run_args = [(i, image_names[i][0], vgg16) for i in run_indices]
	
	if run_args:
		print(f"Generation starting, total {len(run_indices)} images to be processed...")
		
		# without multiprocessing
		# for args in run_args:
		# 	get_image_feature(args)

		# with multiprocessing
		p = multiprocessing.Pool(num_workers)
		with p:
			p.map(get_image_feature, run_args)
	else:
		print("All features generated already.")

	
	features = [np.load(f'../temp/deep_features/{i}.npy') for i in range(num_train_data)]
	features = np.vstack(features)
	print("Features loaded.")
	
	np.savez('trained_system_deep.npz', features=features, labels=labels)
	print("System saved to trained_system_deep.npz.")

def evaluate_recognition_system(vgg16,num_workers=2):
	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8,8)
	* accuracy: accuracy of the evaluated system
	'''

	test_data = np.load("../data/test_data.npz", allow_pickle=True)
	test_paths = test_data['image_names']
	test_labels = test_data['labels']
	trained_system_deep = np.load("trained_system_deep.npz")
	features = trained_system_deep['features']
	labels = trained_system_deep['labels']

	num_train_data = test_paths.shape[0]

	force_run = True
	if force_run:
		run_indices = list(range(num_train_data))
	else:
		run_indices = [i for i in range(num_train_data) if not os.path.exists(f'../temp/deep_features/{i+10000}.npy')]
	run_args = [(i+10000, test_paths[i][0], vgg16) for i in run_indices]
	
	if run_args:
		print(f"Generation starting, total {len(run_indices)} images to be processed...")
		
		# without multiprocessing
		# for args in run_args:
		# 	get_image_feature(args)

		# with multiprocessing
		p = multiprocessing.Pool(num_workers)
		with p:
			p.map(get_image_feature, run_args)
	else:
		print("All features generated already.")

	test_features = [np.load(f'../temp/deep_features/{i+10000}.npy') for i in range(num_train_data)]
	test_features = np.vstack(test_features)
	print("Test dataset features loaded.")

	test_pred = np.array([labels[np.argmin(distance_to_set(feature, features)).item()] for feature in test_features])
	num_classes = max(np.max(test_pred), np.max(test_labels))
	num_classes = int(np.max(test_labels) + 1)
	confusion_matrix = np.zeros((num_classes, num_classes))
	for pred in range(num_classes):
		for truth in range(num_classes):
			confusion_matrix[truth, pred] = np.sum(np.logical_and(test_pred == pred, test_labels == truth))
	accuracy = np.sum(np.diagonal(confusion_matrix))/test_pred.shape[0]
	return confusion_matrix, accuracy

def preprocess_image(image):
	'''
	Preprocesses the image to load into the prebuilt network.

	[input]
	* image: numpy.ndarray of shape (H,W,3)

	[output]
	* image_processed: torch.Tensor of shape (3,H,W)
	'''

	# :D
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

	# VGG-16 assumes that all input imagery to the network 
	# is resized to 224×224 with the three color channels preserved
	# (use skimage.transform.resize() to do this before passing any 
	# imagery to the network). 
	resized_image = skimage.transform.resize(image, (224, 224))

	# And be sure to normalize the image using suggested 
	# mean and std before extracting the feature:
	mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis, :]
	std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis, :]
	normalized_image = (resized_image - mean) / std

	# reshape
	tranposed_image = normalized_image.transpose((2, 0, 1))
	return torch.from_numpy(tranposed_image)

def get_image_feature(args):
	'''
	Extracts deep features from the prebuilt VGG-16 network.
	This is a function run by a subprocess.
 	[input]
	* i: index of training image
	* image_path: path of image file
	* vgg16: prebuilt VGG-16 network.
	* time_start: time stamp of start time
 	[saved]
	* feat: evaluated deep feature
	'''

	i, image_path, vgg16 = args

	print(f"Feature generating for the {i}-th image.")
	image = skimage.io.imread(f"../data/{image_path}")
	image = image.astype('float')/255

	# these two lines basically replaced my whole implementation
	# in network_layers 🥹
	x: torch.Tensor = preprocess_image(image)
	y: torch.Tensor = vgg16.classifier[:4](vgg16.features(x).flatten()) 
	feat = y.detach().numpy()
	np.save(f'../temp/deep_features/{i}.npy', feat)

def distance_to_set(feature,train_features):
	'''
	Compute distance between a deep feature with all training image deep features.

	[input]
	* feature: numpy.ndarray of shape (K)
	* train_features: numpy.ndarray of shape (N,K)

	[output]
	* dist: numpy.ndarray of shape (N)
	'''
	
	# import visual_recog
	# return visual_recog.distance_to_set(feature, train_features)

	# copying instead of modularizing code gives me emotional damage 🥹
	assert feature.shape[0] == train_features.shape[1], f"{(feature.shape[0], train_features.shape[1])}"
	dist = np.sqrt(np.sum((feature[np.newaxis, :] - train_features) ** 2, axis=1))
	return dist
	
