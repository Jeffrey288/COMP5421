import numpy as np
import torchvision
import util
import matplotlib.pyplot as plt
import visual_words
import visual_recog
import deep_recog
import skimage.io

if __name__ == '__main__':

	num_cores = util.get_num_CPU()

	import os
	for path in ["../temp", "../temp/visual_words", "../temp/histograms", "../temp/histograms_test", "../temp/deep_features"]:
		try:
			if not os.path.exists(path):
				os.makedirs(path)
		except:
			pass
	
	print("-=-=- Task 1 -=-=-")

	# path_img = "../data/kitchen/sun_aasmevtpkslccptd.jpg"; filename = "../writeup/q1,3_sun_aasmevtpkslccptd.png"
	# path_img = "../data/desert/sun_aawnvdsxsoudsdwo.jpg"; filename = "../writeup/q1,3_sun_aawnvdsxsoudsdwo.png"
	# path_img = "../data/highway/sun_acpvugnkzrliaqir.jpg"; filename = "../writeup/q1,3_sun_acpvugnkzrliaqir.png"
	path_img = "../data/waterfall/sun_aecgdxztcovcpyvx.jpg"; filename = "../writeup/q1,3_sun_aecgdxztcovcpyvx.png"
	image = skimage.io.imread(path_img)
	image = image.astype('float')/255

	# # q1.1
	# filter_responses = visual_words.extract_filter_responses(image)
	# util.display_filter_responses(filter_responses)

	# # q1.2
	visual_words.compute_dictionary(num_workers=num_cores)
	
	# # q1.3
	# dictionary = np.load('dictionary.npy')
	# wordmap = visual_words.get_visual_words(image,dictionary)
	# util.save_wordmap(wordmap, filename)

	print("-=-=- Task 2 -=-=-")

	# q2.1 
	# hist = visual_recog.get_feature_from_wordmap(wordmap,visual_words.K)
	# plt.stairs(hist, np.arange(visual_words.K + 1))
	# plt.show()

	# q2.2
	# hist_all = visual_recog.get_feature_from_wordmap_SPM(wordmap,visual_recog.layer_num,visual_words.K)
	# print(hist_all.shape, np.sum(hist_all))
	# plt.stairs(hist_all, np.arange(visual_words.K * ((4 ** (visual_recog.layer_num) - 1) // 3) + 1))
	# plt.show()

	# # q2.4
	visual_recog.build_recognition_system(num_workers=num_cores)

	# q2.5
	conf, accuracy = visual_recog.evaluate_recognition_system(num_workers=num_cores)
	print(conf)
	print(np.diag(conf).sum()/conf.sum())

	print("-=-=- Task 3 -=-=-")

	# q3.1, not sure why we have to implement this but sure okay
	# import network_layers
	# vgg16_weights = util.get_VGG16_weights()
	# features = network_layers.extract_deep_feature(image, vgg16_weights)
	# print(features)

	# q3.2
	vgg16 = torchvision.models.vgg16(pretrained=True).double()
	vgg16.eval()
	deep_recog.build_recognition_system(vgg16,num_workers=num_cores//2)
	conf, accuracy = deep_recog.evaluate_recognition_system(vgg16,num_workers=num_cores//2)
	print(conf)
	print(np.diag(conf).sum()/conf.sum())

