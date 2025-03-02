import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import SubtractDominantMotion

# write your script here, we recommend the above libraries for making your animation
def display_frame(frame, mask, show=False):
    plt.figure()
    plt.imshow(frame, cmap='gray')
    plt.imshow(np.zeros_like(frame), cmap='jet', alpha=0.5*mask.astype(float))
    if show:
        plt.show()

frames = np.load('../data/aerialseq.npy')
imH, imW, num_frames = frames.shape
# num_frames = 30
# print(frames.shape)

# display the first frame
# display_frame(frames[:, :, 0], np.zeros((imH, imW)))

# Evaluate the Lucas-Kanade
masks = [np.zeros((imH, imW))]
for frame_idx in range(1, num_frames):
    if (frame_idx - 1) % 1 == 0: print(f'Processing frame {frame_idx}...')
    mask = SubtractDominantMotion.SubtractDominantMotion(frames[:,:,frame_idx-1], frames[:,:,frame_idx])
    masks.append(mask)
    if frame_idx % 30 == 0:
        display_frame(frames[:, :, frame_idx], mask)
        plt.savefig(f'../writeup/q3,3_frame{frame_idx}')

# To view the animation, uncomment this part
fig = plt.figure()
ax = plt.gca()
im = plt.imshow(frames[:, :, 0], cmap='gray', animated=True)
def animate(frame_idx):
    global frames
    img = np.repeat(frames[:,:,frame_idx][:,:,np.newaxis], 3, axis=2)
    img[:, :, 2] = np.minimum(img[:, :, 2] + 0.35 * masks[frame_idx], 1.0)    
    im.set_array(img)
    return im,
anim = animation.FuncAnimation(fig, animate, frames=num_frames, blit=True, interval=1000//60)
plt.show()
