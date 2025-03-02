import numpy as np
import threading
import queue
import imageio
import os,time
import math
import visual_words
import skimage.io

layer_num = 3
def build_recognition_system(num_workers=2):
	'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N,M)
	* labels: numpy.ndarray of shape (N)
	* dictionary: numpy.ndarray of shape (K,3F)
	* SPM_layer_num: number of spatial pyramid layers
	'''

	def workerThread(q: queue.Queue):
		while not q.empty():
			args = q.get()
			generate_histogram(args)
			q.task_done()

	def generate_histogram(args):
		i, file_path = args
		print(f"Generating histogram for the {i}-th image...")
		histogram = get_image_feature(file_path, dictionary, layer_num, visual_words.K)
		np.save(f'../temp/histograms/{i}.npy', histogram)

	train_data = np.load("../data/train_data.npz", allow_pickle=True)
	dictionary = np.load("dictionary.npy")

	labels = train_data['labels']
	image_names = train_data['image_names']
	num_train_data = image_names.shape[0]

	force_run = True
	if force_run:
		run_indices = list(range(num_train_data))
	else:
		run_indices = [i for i in range(num_train_data) if not os.path.exists(f'../temp/histograms/{i}.npy')]
	run_args = [(i, image_names[i][0]) for i in run_indices]
	
	# https://medium.datadriveninvestor.com/the-most-simple-explanation-of-threads-and-queues-in-python-cbc206025dd1
	if run_args:
		print(f"Generation starting, total {len(run_indices)} images to be processed...")

		# without threading
		# for args in run_args:
		# 	generate_histogram(args)

		# with threading
		q = queue.Queue()
		for args in run_args:
			q.put(args)
		for _ in range(num_workers):
			t = threading.Thread(target=workerThread, args=(q, ), daemon=True)
			t.start()
		q.join()
		print("Generation completed.")
	else:
		print("All wordmaps generated already.")

	features = [np.load(f'../temp/histograms/{i}.npy') for i in range(num_train_data)]
	features = np.vstack(features)
	print("Features loaded.")

	np.savez('trained_system.npz',
		dictionary = dictionary,
		features = features,
		labels = labels,
		SPM_layer_num = layer_num)
	print("System successfully saved to trained_system.npz.")
	
def evaluate_recognition_system(num_workers=2):
	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8,8)
	* accuracy: accuracy of the evaluated system
	'''

	test_data = np.load("../data/test_data.npz", allow_pickle=True)
	test_paths = test_data['image_names']
	test_labels = test_data['labels']
	trained_system = np.load("trained_system.npz")

	dictionary = trained_system['dictionary']
	features = trained_system['features']
	labels = trained_system['labels']
	layer_num = trained_system['SPM_layer_num']

	def workerThread(q: queue.Queue):
		while not q.empty():
			args = q.get()
			generate_histogram(args)
			q.task_done()

	def generate_histogram(args):
		i, file_path = args
		print(f"Generating histogram for the {i}-th image...")
		histogram = get_image_feature(file_path, dictionary, layer_num, visual_words.K)
		np.save(f'../temp/histograms_test/{i}.npy', histogram)

	num_train_data = test_paths.shape[0]

	force_run = True
	if force_run:
		run_indices = list(range(num_train_data))
	else:
		run_indices = [i for i in range(num_train_data) if not os.path.exists(f'../temp/histograms_test/{i}.npy')]
	run_args = [(i, test_paths[i][0]) for i in run_indices]
	
	# https://medium.datadriveninvestor.com/the-most-simple-explanation-of-threads-and-queues-in-python-cbc206025dd1
	# the documentation for threading and queues are acutally very clear
	if run_args:
		print(f"Generation starting, total {len(run_indices)} images to be processed...")

		# without threading
		# for args in run_args:
		# 	generate_histogram(args)

		# with threading
		q = queue.Queue()
		for args in run_args:
			q.put(args)
		for _ in range(num_workers):
			t = threading.Thread(target=workerThread, args=(q, ), daemon=True)
			t.start()
		q.join()
		print("Generation completed.")
	else:
		print("All wordmaps generated already.")

	test_features = [np.load(f'../temp/histograms_test/{i}.npy') for i in range(num_train_data)]
	print("All wordmaps loaded.")
	test_pred = np.array([labels[np.argmax(distance_to_set(feature, features)).item()] for feature in test_features])
	num_classes = max(np.max(test_pred), np.max(test_labels))
	num_classes = int(np.max(test_labels) + 1)
	confusion_matrix = np.zeros((num_classes, num_classes))
	for pred in range(num_classes):
		for truth in range(num_classes):
			confusion_matrix[truth, pred] = np.sum(np.logical_and(test_pred == pred, test_labels == truth))
	accuracy = np.sum(np.diagonal(confusion_matrix))/test_pred.shape[0]

	# q2.6: visual a few failed cases
	# import matplotlib.pyplot as plt
	# failed_ind = np.argwhere(test_pred != test_labels).reshape((-1, ))
	# label_names = ["auditorium", "baseball_field", "desert", "highway", "kitchen", "laundromat", "waterfall", "windmill"]
	# for i in failed_ind:
	# 	image = skimage.io.imread(f"../data/{test_paths[i][0]}")
	# 	plt.imshow(image)
	# 	plt.title(f"Actual: {label_names[test_labels[i]]}, Predicted: {label_names[test_pred[i]]}")
	# 	plt.suptitle(f"Image: {test_paths[i][0]}")
	# 	plt.savefig(f"../writeup/q2,6_case_{i}.png")
	# 	if i % 5 == 0: plt.show()
	
	return confusion_matrix, accuracy	
		
def get_image_feature(file_path,dictionary,layer_num,K):
	'''
	Extracts the spatial pyramid matching feature.

	[input]
	* file_path: path of image file to read
	* dictionary: numpy.ndarray of shape (K,3F)
	* layer_num: number of spatial pyramid layers
	* K: number of clusters for the word maps

	[output]
	* feature: numpy.ndarray of shape (K)
	'''
	image = skimage.io.imread(f"../data/{file_path}")
	image = image.astype('float')/255

	wordmap = visual_words.get_visual_words(image, dictionary)
	feature = get_feature_from_wordmap_SPM(wordmap, layer_num, K)
	return feature 

def distance_to_set(word_hist,histograms):
	'''
	Compute similarity between a histogram of visual words with all training image histograms.

	[input]
	* word_hist: numpy.ndarray of shape (K)
	* histograms: numpy.ndarray of shape (N,K)

	[output]
	* sim: numpy.ndarray of shape (N)
	'''
	assert word_hist.shape[0] == histograms.shape[1], f"{(word_hist.shape[0], histograms.shape[1])}"
	return np.sum(np.minimum(word_hist[np.newaxis, :], histograms), axis=1)
	
def get_feature_from_wordmap(wordmap,dict_size):
	'''
	Compute histogram of visual words.

	[input]
	* wordmap: numpy.ndarray of shape (H,W)
	* dict_size: dictionary size K

	[output]
	* hist: numpy.ndarray of shape (K)
	'''
	return np.bincount(wordmap.reshape((-1, )), minlength=dict_size) / (wordmap.shape[0] * wordmap.shape[1])
	
def get_feature_from_wordmap_SPM(wordmap,layer_num,dict_size):
	'''
	Compute histogram of visual words using spatial pyramid matching.

	[input]
	* wordmap: numpy.ndarray of shape (H,W)
	* layer_num: number of spatial pyramid layers
	* dict_size: dictionary size K

	[output]
	* hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
	'''

	hist_all = [[] for _ in range(layer_num)]

	h_slices = [wordmap.shape[0] * i // (2 ** (layer_num - 1)) for i in range(2 ** (layer_num - 1) + 1)] 
	w_slices = [wordmap.shape[1] * i // (2 ** (layer_num - 1)) for i in range(2 ** (layer_num - 1) + 1)] 
	# first, generate the finest layer
	for h in range(2 ** (layer_num - 1)):
		for w in range(2 ** (layer_num - 1)):
			hist_all[layer_num - 1].append(4.0 ** (1 - layer_num) * get_feature_from_wordmap(wordmap[h_slices[h]:h_slices[h+1], w_slices[w]:w_slices[w+1]], dict_size))

	# then, we sum the previous areas to get the upper layers
	for l in range(layer_num - 2, -1, -1):
		for h in range(0, 2 ** (l + 1), 2):
			for w in range(0, 2 ** (l + 1), 2):
				hist_all[l].append(sum([hist_all[l+1][w+wi+2**(l+1)*(h+hi)] for wi in range(2) for hi in range(2)]))

	# we concatenate all layers to get our final histogram
	# print([[np.sum(x) for x in y] for y in hist_all])
	hist_all = [[wordmap * (2.0**(max(1, l)-layer_num)) for wordmap in hist_all[l]] for l in range(layer_num)]
	# print([[np.sum(x) for x in y] for y in hist_all])
	hist_all = np.hstack(sum(hist_all, []))
	return hist_all






	

