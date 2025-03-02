def display_frame(frame, rect, rect_baseline = None, show=False):
    plt.figure()
    plt.imshow(frame, cmap='gray')
    plt.gca().add_patch(get_patch(rect))
    if rect_baseline is not None:
        plt.gca().add_patch(get_patch(rect_baseline, color="green"))
    if show:
        plt.show()