spts1 = [[371, 289], [194, 269], [316, 221], [150, 220], [172, 325], [380, 292], [255, 203]] 
spts2 = [[373, 283], [194, 258], [315, 200], [149, 195], [173, 329], [383, 287], [254, 187]] 
F = [[-4.9394929404270995e-09, 2.113551574035296e-06, -0.0005297279319451781], [-2.63170538779425e-06, 1.6199947800382637e-08, 0.0006277697979742132], [0.0006951164519825967, -0.0005076505150703298, -0.03936932769129756]]

import numpy as np
hpts1 = np.hstack([spts1, np.ones((7, 1))])
hpts2 = np.hstack([spts2, np.ones((7, 1))])
F = np.array(F)

# print(hpts1 @ F)
errs = []
for i in range(7):
    print(hpts2[i, np.newaxis, :], F, hpts1[i, :, np.newaxis])
    err = hpts2[i, np.newaxis, :] @ F @ hpts1[i, :, np.newaxis]
    errs.append(np.abs(err[0, 0]))
errs = np.array(errs)
# g = np.sum((hpts2 @ F) * hpts1, axis=1)
g = np.sum(hpts2 * (hpts1 @ F.T), axis=1)
print(errs, g)