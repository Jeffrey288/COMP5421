import numpy as np
import matplotlib.pyplot as plt

def alignChannels(red, green, blue):
    """Given 3 images corresponding to different channels of a color image,
    compute the best aligned result with minimum abberations

    Args:
      red, green, blue - each is a HxW matrix corresponding to an HxW image

    Returns:
      rgb_output - HxWx3 color image output, aligned as desired"""

    """
    Logic:
    first align red and green, then align green and blue
    calculated average SSD across overlapped area

    +x, +y means latter photo is lower / righter than the former photo
    """


    def avgSSD(u, v):
        return np.sum(np.square(u - v), axis=(0, 1)) / (u.shape[0] * u.shape[1])
    def shift_crop(u, v, x, y):
        return u[(y if y > 0 else None):(y if y < 0 else None), (x if x > 0 else None):(x if x < 0 else None)], \
            v[(-y if y < 0 else None):(-y if y > 0 else None), (-x if x < 0 else None):(-x if x > 0 else None)]

    offsetArray = [[[x, y] for x in range(-30, 30+1)] for y in range(-30, 30+1)]
    rgSSD = np.array([[avgSSD(*shift_crop(red, green, x, y)) for x in range(-30, 30+1)] for y in range(-30, 30+1)])
    rgMinInd = np.where(rgSSD == np.min(rgSSD)); rgMinInd = list(map(int, rgMinInd))
    rgOffset = np.array(offsetArray[rgMinInd[0]][rgMinInd[1]])
    gbSSD = np.array([[avgSSD(*shift_crop(green, blue, x, y)) for x in range(-30, 30+1)] for y in range(-30, 30+1)])
    gbMinInd = np.where(gbSSD == np.min(gbSSD)); gbMinInd = list(map(int, gbMinInd))
    gbOffset = np.array(offsetArray[gbMinInd[0]][gbMinInd[1]])
    # rgOffset = np.array([-3, -12]) 
    # gbOffset = np.array([-5, -9])
    # print(rgOffset, gbOffset)

    gCoords = rgOffset[:]
    bCoords = rgOffset + gbOffset
    shift = {'x': (0, gCoords[0], bCoords[0]), 'y': (0, bCoords[1], gCoords[1])}
    minShift = np.array([min(shift["x"]), min(shift["y"])])
    new_img = np.zeros((red.shape[0] - min(shift["y"]) + max(shift["y"]),  \
                        red.shape[1] - min(shift["x"]) + max(shift["x"]), 3), dtype=np.uint8)
    gCoords -= minShift; bCoords -= minShift
    new_img[-minShift[1]:-minShift[1] + red.shape[0], -minShift[0]:-minShift[0] + red.shape[1], 0] = red
    new_img[gCoords[1]:gCoords[1] + green.shape[0], gCoords[0]:gCoords[0] + green.shape[1], 1] = green
    new_img[bCoords[1]:bCoords[1] + blue.shape[0], bCoords[0]:bCoords[0] + blue.shape[1], 2] = blue
    # plt.imshow(new_img)
    # plt.show()

    # cropped = shift_crop(red, green, rgOffset[1], rgOffset[0])
    # print(red.shape, cropped[0].shape, cropped[1].shape)
    # f, ax = plt.subplots(1, 2)
    # ax[0].imshow(cropped[0])
    # ax[1].imshow(cropped[1])
    # plt.show()

    return new_img
