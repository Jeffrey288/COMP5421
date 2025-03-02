en/sun_aasmevtpkslccptd.jpg"; filename = "../writeup/q1,3_sun_aasmevtpkslccptd.png"
	# path_img = "../data/desert/sun_aawnvdsxsoudsdwo.jpg"; filename = "../writeup/q1,3_sun_aawnvdsxsoudsdwo.png"
	# path_img = "../data/highway/sun_acpvugnkzrliaqir.jpg"; filename = "../writeup/q1,3_sun_acpvugnkzrliaqir.png"
	# path_img = "../data/waterfall/sun_aecgdxztcovcpyvx.jpg"; filename = "../writeup/q1,3_sun_aecgdxztcovcpyvx.png"
	image = skimage.io.imread(path_img)
	image = image.astype('float')/255
