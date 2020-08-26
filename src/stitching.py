""" Stitch image

Author: 
Date:February 29, 2020
"""

# module
import pickle
import cv2
import numpy as np
from utils import homogenize, dehomogenize
from rendering import color_map_local
# parameters


# classes


# functions

def color_map(map_local):
    """ color the map by which label has max number of points """
    d = 0.1
    boundary = [[-20, 50], [-10, 10]]
    catogories = [128, 140, 255, 107, 244] # values of labels in the iuput images of mapping
    catogories_color = np.array([
        [128, 64, 128], # road
        [140, 140, 200], # crosswalk
        [255, 255, 255], # lane
        [107, 142, 35], # vegetation
        [244, 35, 232] # sidewalk
    ])
    return color_map_local(map_local, catogories, catogories_color)

def read_pickle(filename):    
    with open(filename, "rb") as filehandler:
        picked_file = pickle.load(filehandler)
        return picked_file

def stitch_image(im_src_list, homography_list, log_odds_out=True):
    
    

    imSize = im_src_list[0].shape
    anchor = np.array([
        [imSize[1], 0, 0, imSize[1]],
        [0, 0, imSize[0], imSize[0]]
    ])

    x = homogenize(anchor)
    x_t = np.array(x)
    h_t = np.eye(3)

    min_x, min_y = 0, 0
    max_x, max_y = imSize[1], imSize[0]
    for h in homography_list[::-1]:
        x_t = np.matmul(h, x_t)
        h_t = np.matmul(h, h_t)
        print(h[0,2], h[1,2])
        min_x_t = np.min(x_t[0,:])
        min_y_t = np.min(x_t[1,:])
        max_x_t = np.max(x_t[0,:])
        max_y_t = np.max(x_t[1,:])
        if min_x_t < min_x:
            min_x = min_x_t
        if min_y_t < min_y:
            min_y = min_y_t
        if max_x_t > max_x:
            max_x = max_x_t
        if max_y_t > max_y:
            max_y = max_y_t

    max_x, max_y, min_x, min_y = int(max_x), int(max_y), int(min_x), int(min_y)
    x_dst = dehomogenize(x_t)
    out_size = [max_x - min_x, max_y - min_y]
    if log_odds_out:
        channel = im_src_list[0].shape[2]
        im_dst = np.zeros((out_size[1], out_size[0], channel))
    else:
        im_dst = np.zeros((out_size[1], out_size[0], 3)).astype(np.uint8)
    
    for i in range(len(homography_list)-1):
        x_t = np.array(x)
        h_t = np.eye(3)
        for h in homography_list[i:-1]:
            h_t = np.matmul(h, h_t)
        h_t[0,2] -= min_x
        h_t[1,2] -= min_y
        if log_odds_out:
            im_src = im_src_list[i]
        else:
            im_src = color_map(im_src_list[i])
        im_out = cv2.warpPerspective(im_src, h_t, (out_size[0], out_size[1]))
        if log_odds_out:
            im_dst += im_out
        else:
            mask = np.sum(im_out, axis=2) != 0
            im_dst[mask] = im_out[mask]
    
    return im_dst

# main
def main():
    filename = "/home/henry/log_odds_1.pickle"
    dict_in = read_pickle(filename)
    im_src_list = dict_in['log_odds']
    homography_list = dict_in['h']
    log_odds_out = True
    im_dst = stitch_image(im_src_list[18:28], homography_list[18:28], log_odds_out=log_odds_out)
    if log_odds_out:
        im_out = color_map(im_dst)
    else:
        im_out = im_dst
    cv2.imshow("im", im_out)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()