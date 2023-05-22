#!/usr/bin/env python
# coding: utf-8

# In[56]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from collections import Counter
import sys
import time
import copy
import rospy

def generate_convex_hull(img_src, vis=False, index_care_about=1, index_to_vitualize=None, top_number=1, area_threshold=30):

    """
        Generate the convex hull
        Args:
            img: input img (h, w, 3)
            vis: True if vitualize the result; Only vitualize the last convex hull
            index_care_about: index that will be used to generate the convex hull
            index_to_vitualize: index that will be used to vitualize the result (index of the convex hull)
            top_number: the number most common label decided to choose
            area_threshold: only consider the connected component which contains the points greater than this area_threshold
        Returns:
            vertices: extracted vertices; list of numpy arrays; array shape- -- [2, number of vertices]
    """
    rows, cols = img_src.shape
    img = np.array(img_src)
    if index_care_about == 0:
        rospy.logerr("index care about cannot be zero in this version of code")
        exit(0)
    img[img[:,:]!=index_care_about] = 0
    img[img[:,:]==index_care_about] = 1
    vertices = []
    
    kernel = np.ones((3,3), np.uint8)
    crosswalk = np.copy(img[:,:])
    if vis == True:
        plt.figure(0)
        plt.imshow(crosswalk)
    crosswalk = cv2.erode(crosswalk, kernel, iterations=1)

    if vis == True:
        plt.figure(1)
        plt.imshow(crosswalk)

    crosswalks = label(crosswalk, connectivity=crosswalk.ndim)

    if np.all(crosswalks==0):
        return []
    if vis == True:
        plt.figure(2)
        plt.imshow(crosswalks)
    if index_to_vitualize == None:
        count = Counter(crosswalks[crosswalks!=0].reshape(-1)).most_common(top_number)
        index_to_vitualize = [x[0] for x in count if x[1] > area_threshold]

    for select_index in index_to_vitualize:
        chosen_crosswalk = np.copy(crosswalks)
        crosswalk_pts = np.zeros((1,2))
        indexes = np.where(chosen_crosswalk==select_index)

        crosswalk_pts = np.concatenate([np.array([i,j]).reshape(1,2) for (i,j) in zip(*indexes)])
        # Here I modefy comment the next line. And calculate all the vertices for 
        # chosen_crosswalk[chosen_crosswalk!=select_index] = 9

        crosswalk_pts = crosswalk_pts[1:, :]
        crosswalk_pts = np.fliplr(crosswalk_pts)

        hull = cv2.convexHull(crosswalk_pts)
        nodes = np.concatenate([np.squeeze(hull), hull[0,:,:].reshape(1,-1)],axis=0).T
        vertices.append(nodes)
        x_vertices = vertices[-1][0, :]
        y_vertices = vertices[-1][1, :]
    
    if vis == True:
        plt.figure(3)
        plt.imshow(chosen_crosswalk)

        fig = plt.figure(4)
        ax = fig.add_subplot(1,1,1)
        plt.figure(5)
        plt.imshow(img[:,:])   
        plt.scatter(x_vertices, y_vertices, s=50, c='red', marker='o')
        plt.plot(x_vertices, y_vertices, c='red')
        plt.show()
    return vertices

def test_generate_convex_hull():
    import time

    img = cv2.imread('./tempimage.jpg', cv2.IMREAD_GRAYSCALE)

    tic = time.time()
    generate_convex_hull(img, vis=False)
    toc = time.time()
    print("running time: {:.6f}s".format(toc - tic))

def test_generate_convex_hull_segfault():
    rospy.init_node('fake_node')
    arr = np.load('test/debug.npy')
    # print(arr)
    vertices = generate_convex_hull(arr)
    
    # if you do not see printint this, it is because a segmentation fault
    print(vertices)

def main():
    # test_generate_convex_hull()
    test_generate_convex_hull_segfault()

if __name__ == "__main__":
    main()
