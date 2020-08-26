#!/usr/bin/env python
""" Projection the semantic image to the world frame

Author: Henry Zhang
Date:February 18, 2020
"""

# module
import cv2
import numpy as np

from src.utils.utils import dehomogenize


# parameters


# classes


# functions
def generate_homography(im_src, pts_src, pts_dst, vis=False, out_size=None, return_h=False):
    """ generate homography transformed image from point corrspondences

    Params:
        im_src: input image
        pts_src: n by 2 array
        pts_dst: n by 2 array
        out_size: [width, height] specify the output size
    Return:
        im_dst: transformed image
    link:
    https://www.learnopencv.com/homography-examples-using-opencv-python-c/
    """
    # Calculate Homography
    assert (len(pts_src[1]) == 2)
    assert (len(pts_dst[1]) == 2)

    h, status = cv2.findHomography(pts_src, pts_dst)

    # Calculate shape
    shape_pts = np.array(
        [[0, 0, 1], [0, im_src.shape[0], 1], [im_src.shape[1], im_src.shape[0], 1], [im_src.shape[1], 0, 1.]])
    shape_pts_dst = np.matmul(h, shape_pts.T)
    shape_pts_dst = dehomogenize(shape_pts_dst)
    size_dst_max = (np.max(shape_pts_dst, axis=1)).astype(np.int)  # -
    size_dst_min = (np.min(shape_pts_dst, axis=1)).astype(np.int)
    # if np.min(size_dst_min) < 0:
    #     raise NotImplementedError

    # Warp source image to destination based on homography
    if out_size is None:
        im_dst = cv2.warpPerspective(im_src, h, (im_src.shape[1], im_src.shape[0]))  #
    else:
        im_dst = cv2.warpPerspective(im_src, h, (out_size[0], out_size[1]))  #
    # im_dst = cv2.warpPerspective(im_src, h, (size_dst_max[0],size_dst_max[1])) #
    # im_dst = im_dst[size_dst_min[1]::, size_dst_min[0]::]
    if vis == True:

        # Display images
        pts_src = pts_src.reshape((-1, 1, 2)).astype(np.int32)
        im_src = cv2.polylines(im_src, [pts_src], False, (0, 255, 255))
        if im_src.shape[1] > 1000:
            im_src = cv2.resize(im_src, (800, 600))
        cv2.imshow("Source Image", im_src)
        # cv2.imshow("Destination Image", im_dst)
        pts_dst = pts_dst.reshape((-1, 1, 2)).astype(np.int32)
        im_dst = cv2.polylines(im_dst, [pts_dst], False, (0, 255, 255))
        cv2.imshow("Warped Source Image", im_dst)

        cv2.waitKey(0)

    if return_h:
        return im_dst, h
    else:
        return im_dst


def test_generate_homography():
    # Read source image.
    im_src = cv2.imread(
        '/home/henry/Documents/projects/pylidarmot/src/vision_semantic_segmentation/network_output_example/preds/2790.jpg')
    im_src = int(255 / np.max(im_src)) * im_src

    # Four corners of the book in source image
    pts_src = np.array([[141., 131], [480, 159], [493, 630], [64, 601]], np.int32)

    # Read destination image.
    # im_dst = cv2.imread('book1.jpg')
    # Four corners of the book in destination image.
    pts_dst = np.array([[318., 256], [534, 372], [316, 670], [73, 473]], np.int32)
    generate_homography(im_src, pts_src, pts_dst, vis=True)


# main
def main():
    test_generate_homography()


if __name__ == "__main__":
    main()
