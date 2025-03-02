'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import numpy as np
import matplotlib.pyplot as plt
import submission as sub
import helper

data = np.load('../data/templeCoords.npz')
x1 = data['x1']
y1 = data['y1']
pts1 = np.hstack([x1, y1]).T

with np.load('../data/intrinsics.npz') as intrinsics:
    K1 = intrinsics['K1']
    K2 = intrinsics['K2']
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

N = x1.shape[0]
M = 640

# first we hardcode the fundamental matrix we got
# from q2.1 :D
F = np.array([[-8.331492341800977e-09, 1.2953846201515214e-07, -0.0011718785098119202], [6.513583362032381e-08,
             5.706700587341282e-09, -4.134350366704326e-05], [0.0011307876458092711, 1.918236366032668e-05, 0.004168620793724458]])

# q4.1
# helper.epipolarMatchGUI(im1, im2, F)

# q4.2
x2 = []
y2 = []
for i in range(N):
    x, y = sub.epipolarCorrespondence(im1, im2, F, x1[i, 0], y1[i, 0])
    x2.append(x)
    y2.append(y)
x2 = np.vstack(x2)
y2 = np.vstack(y2)

M1 = np.hstack([np.eye(3), np.zeros((3, 1))])

# hardcode the obtained matrix
M2 = np.array(
    [[0.9993695037930264, 0.035197572534239496, -0.004661091736671946, 0.01780295850034071], [-0.03283857784245489, 0.9662338863618545, 0.255565460599339, -1.0], [0.013498988620105281, -0.25525126392197206, 0.9667805177869844, 0.08705061683747736]]
)

C1 = K1 @ M1
C2 = K2 @ M2

print(K1, K2)
print(M1, M2)

print("d", C1, C2)

P, err = sub.triangulate(C1, np.hstack([x1, y1]), C2, np.hstack([x2, y2]))
print(err)

def plotP(P):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], marker="o")
    plt.show()

plotP(P)

np.savez('q4_2.npz', F=F, M1=M1, M2=M2, C1=C1, C2=C2)
