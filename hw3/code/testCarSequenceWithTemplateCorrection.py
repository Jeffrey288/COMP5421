import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import LucasKanade

# write your script here, we recommend the above libraries for making your animation
def get_patch(rect, color="yellow"):
    return patches.Rectangle(rect[:2], rect[2]-rect[0], rect[3]-rect[1], fill=False, edgecolor=color)
def display_frame(frame, rect, rect_baseline = None, show=False):
    plt.figure()
    plt.imshow(frame, cmap='gray')
    plt.gca().add_patch(get_patch(rect))
    if rect_baseline is not None:
        plt.gca().add_patch(get_patch(rect_baseline, color="green"))
    if show:
        plt.show()

frames = np.load('../data/carseq.npy')
imH, imW, num_frames = frames.shape
# num_frames = 50
print(frames.shape)

rects = np.zeros((num_frames, 4), dtype=float)
rects[0] = [59.0, 116.0, 145.0, 151.0] # we are doing subpixel interpolation

# baseline comparison
rects_q13 = np.zeros((num_frames, 4), dtype=float)
rects_q13[0] = [59.0, 116.0, 145.0, 151.0]

# display the first frame
display_frame(frames[:, :, 0], rects[0], rects_q13[0])
plt.savefig('../writeup/q1,4_frame0')

# Evaluate the Lucas-Kanade
fig = plt.figure()
p_prev = [0, 0]
for frame_idx in range(1, num_frames):
    if frame_idx % 10 == 0: print(f'Processing frame {frame_idx}...')
    rect = rects[frame_idx-1]
    # when talking about p, we are referring to the *accumulative* deviation from the first frame
    p_new = p_prev + LucasKanade.LucasKanade(frames[:,:,frame_idx-1], frames[:,:,frame_idx], rect) 
    p_star = LucasKanade.LucasKanade(frames[:,:,0], frames[:,:,frame_idx], rects[0], p0=p_new) 
    # print(np.linalg.norm(p_star - p_new), p_star, p_new)
    if np.linalg.norm(p_star - p_new) < 2: # will lose track of car if this number is too small
        p_step = p_star - p_prev
        rects[frame_idx] = [rect[0] + p_step[0], rect[1] + p_step[1], rect[2] + p_step[0], rect[3] + p_step[1]]
        p_prev = p_star
    else:
        rects[frame_idx] = rects[frame_idx-1]

    rect_q13 = rects_q13[frame_idx-1]
    p_q13 = LucasKanade.LucasKanade(frames[:,:,frame_idx-1], frames[:,:,frame_idx], rect_q13)
    rects_q13[frame_idx] = [rect_q13[0] + p_q13[0], rect_q13[1] + p_q13[1], rect_q13[2] + p_q13[0], rect_q13[3] + p_q13[1]]

    if frame_idx % 100 == 0:
        display_frame(frames[:, :, frame_idx], rects[frame_idx], rects_q13[frame_idx])
        plt.savefig(f'../writeup/q1,4_frame{frame_idx}')

# Save to rects
np.save('carseqrects-wcrt.npy', rects)

# To view the animation, uncomment this part
fig = plt.figure()
ax = plt.gca()
im = plt.imshow(frames[:, :, 0], cmap='gray', animated=True)
def animate(frame_idx):
    global frames, rects
    print(frame_idx)
    im.set_array(frames[:, :, frame_idx])
    patch = ax.add_patch(get_patch(rects[frame_idx]))
    patch2 = ax.add_patch(get_patch(rects_q13[frame_idx], color="green"))
    return im, patch, patch2
anim = animation.FuncAnimation(fig, animate, frames=num_frames, blit=True, interval=1000//60)
plt.show()
