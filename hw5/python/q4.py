import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    # import matplotlib.pyplot as plt
    # import matplotlib.patches as mpatches

    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions
    gray = skimage.color.rgb2gray(image)

    # threshold
    thresh = skimage.filters.threshold_otsu(gray)
    bw = (gray > thresh).astype(float)
    # print(bw > thresh, bw.shape)
    # plt.imshow((bw > thresh).astype(float), cmap='gray')
    # plt.show()

    # connects separated but close components
    closed = skimage.morphology.closing(1.0 - bw, skimage.morphology.square(8))
    # plt.imshow(closed, cmap='gray')
    # plt.show()

    # remove artifacts connected to image border
    cleared = skimage.segmentation.clear_border(closed)

    # label image regions 
    # for each connected component, set the pixel value as
    # an integer which is its label
    label_image = skimage.measure.label(cleared)
    # plt.imshow(label_image, cmap='gray')
    # plt.show()

    # display the bounding boxes
    # to make the background transparent, pass the value of `bg_label`,
    # and leave `bg_color` as `None` and `kind` as `overlay`
    # image_label_overlay = skimage.color.label2rgb(label_image, image=image, bg_label=0)
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.imshow(image_label_overlay)

    bboxes = []
    for region in skimage.measure.regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 100:
            bboxes.append(region.bbox)
            # draw rectangle around segmented coins
            # minr, minc, maxr, maxc = region.bbox
            # rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
            #                         fill=False, edgecolor='red', linewidth=2)
            # ax.add_patch(rect)

    # ax.set_axis_off()
    # plt.tight_layout()
    # plt.show()

    return bboxes, bw