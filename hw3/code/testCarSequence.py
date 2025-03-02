import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import LucasKanade

# write your script here, we recommend the above libraries for making your animation
def get_patch(rect):
    return patches.Rectangle(rect[:2], rect[2]-rect[0], rect[3]-rect[1], fill=False, edgecolor="yellow")
def display_frame(frame, rect, show=False):
    plt.figure()
    plt.imshow(frame, cmap='gray')
    plt.gca().add_patch(get_patch(rect))
    if show:
        plt.show()

frames = np.load('../data/carseq.npy')
imH, imW, num_frames = frames.shape
print(frames.shape)

rects = np.zeros((num_frames, 4), dtype=float)
rects[0] = [59.0, 116.0, 145.0, 151.0] # we are doing subpixel interpolation

# display the first frame
display_frame(frames[:, :, 0], rects[0])
plt.savefig('../writeup/q1,3_frame0')

# Evaluate the Lucas-Kanade
fig = plt.figure()
for frame_idx in range(1, num_frames):
    if frame_idx % 10 == 0: print(f'Processing frame {frame_idx}...')
    rect = rects[frame_idx-1]
    p = LucasKanade.LucasKanade(frames[:,:,frame_idx-1], frames[:,:,frame_idx], rect)
    rects[frame_idx] = [rect[0] + p[0], rect[1] + p[1], rect[2] + p[0], rect[3] + p[1]]
    if frame_idx % 100 == 0:
        display_frame(frames[:, :, frame_idx], rects[frame_idx])
        plt.savefig(f'../writeup/q1,3_frame{frame_idx}')

# Save to rects
np.save('carseqrects.npy', rects)

# To view the animation, uncomment this part
fig = plt.figure()
ax = plt.gca()
im = plt.imshow(frames[:, :, 0], cmap='gray', animated=True)
def animate(frame_idx):
    global frames, rects
    print(frame_idx)
    im.set_array(frames[:, :, frame_idx])
    patch = ax.add_patch(get_patch(rects[frame_idx]))
    return im, patch
anim = animation.FuncAnimation(fig, animate, frames=num_frames, blit=True, interval=1000//60)
plt.show()

# Without blit, the animation is much slower
# fig = plt.figure()
# ax = plt.gca()
# def animate(frame_idx):
#     global frames, rects
#     ax.clear()
#     ax.set_title(f'Frame {frame_idx}')
#     ax.imshow(frames[:, :, frame_idx], cmap='gray', animated=True)
#     ax.add_patch(get_patch(rects[frame_idx]))
# anim = animation.FuncAnimation(fig, animate, frames=num_frames, blit=False, interval=1000//60) # blit is better lol
# plt.show()