def test_evaluate():
    import cv2
    from matplotlib import pyplot as plt
    prediction = cv2.imread('/home/henry/Pictures/global_map_rendered.png')
    groundtruth = cv2.imread('/home/henry/Downloads/bev.png')

    ## preprocessing
    # resize

    # rotate

    # align

    ## compute metric


    ## visualization
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.canvas.manager.full_screen_toggle()

    ax1.imshow(prediction)
    ax2.imshow(groundtruth)

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.show()

def main():
    test_evaluate()

if __name__ == "__main__":
    main()