#!/usr/bin/env python3
""" Semantic Segmentation Ros Wrapper

Author: Hengyuan Zhang
Date:February 14, 2020
"""

# module
from __future__ import absolute_import, division, print_function, unicode_literals  # python2 compatibility

import argparse
import cv2
import os.path as osp
import rospy
import numpy as np
import sys
import torch

# Add src directory into the path
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "../")))
# Add network directory into the path
sys.path.insert(0, osp.join(osp.dirname(__file__), "network"))

import src.network.deeplab_v3_plus.data.utils.mapillary_visualization as mapillary_visl

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from shape_msgs.msg import Plane
from visualization_msgs.msg import Marker, MarkerArray

from src.camera import camera_setup_6, camera_setup_1
from src.node_config.base_cfg import get_cfg_defaults
from src.network.core.utils.torch_util import set_random_seed
from src.plane_3d import Plane3D
from src.semantic_convex_hull import generate_convex_hull
# from src.semantic_segmentation import SemanticSegmentation  # source code
from src.hrnet.hrnet_semantic_segmentation import HRNetSemanticSegmentation, get_custom_hrnet_args
from src.vis import visualize_marker


# classes
class SemanticSegmentationNode:

    def __init__(self, cfg):
        if cfg.VISION_SEM_SEG.IMAGE_SCALE < 0 or cfg.VISION_SEM_SEG.IMAGE_SCALE > 1:
            raise ValueError("image scale should be in the range of [0, 1]")

        # Set up ros message subscriber
        # Note that topics ".../image_raw" are created by the launch file
        self.image_sub_cam1 = rospy.Subscriber("/camera1/image_raw", Image, self.image_callback, queue_size=1)
        self.image_sub_cam6 = rospy.Subscriber("/camera6/image_raw", Image, self.image_callback, queue_size=1)
        self.plane_sub = rospy.Subscriber("/estimated_plane", Plane, self.plane_callback)

        # Set up ros message publisher
        self.image_pub_cam1 = rospy.Publisher("/camera1/semantic", Image, queue_size=1)
        self.image_pub_cam6 = rospy.Publisher("/camera6/semantic", Image, queue_size=1)
        self.pub_crosswalk_markers = rospy.Publisher("/crosswalk_convex_hull_rviz", MarkerArray, queue_size=10)
        self.pub_road_markers = rospy.Publisher("/road_convex_hull_rviz", MarkerArray, queue_size=10)

        # Load the configuration
        network_cfg = cfg.VISION_SEM_SEG.SEM_SEG_NETWORK
        # self.seg = SemanticSegmentation(network_cfg)
        self.seg = HRNetSemanticSegmentation(get_custom_hrnet_args())
        self.seg_color_fn = mapillary_visl.apply_color_map
        self.seg_color_ref = mapillary_visl.get_labels(network_cfg.DATASET_CONFIG)

        self.plane = None
        self.plane_last_update_time = rospy.get_rostime()
        self.cam6 = camera_setup_6()
        self.cam1 = camera_setup_1()

        self.hull_id = 0
        self.image_scale = cfg.VISION_SEM_SEG.IMAGE_SCALE  # Resize the image to reduce the memory overhead, in percentage.
        self.bridge = CvBridge()

    def image_callback(self, msg):
        rospy.logdebug("Segmented image at: %d.%09ds", msg.header.stamp.secs, msg.header.stamp.nsecs)
        try:
            image_in = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)
            return

        ## ========== Image preprocessing
        image_in = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB)
        if msg.header.frame_id == "camera1":
            image_in = cv2.undistort(image_in, self.cam1.K, self.cam1.dist)
        elif msg.header.frame_id == "camera6":
            image_in = cv2.undistort(image_in, self.cam6.K, self.cam6.dist)
        else:
            rospy.logwarn("unseen camera frame id %s, no undistortion performed.", msg.header.frame_id)

        # Resize the image to reduce the memory overhead
        if self.image_scale < 1:
            width = int(image_in.shape[1] * self.image_scale)
            height = int(image_in.shape[0] * self.image_scale)
            dim = (width, height)
            image_in_resized = cv2.resize(image_in, dim, interpolation=cv2.INTER_AREA)
        else:
            image_in_resized = image_in

        ## ========== semantic segmentation
        image_out_resized = self.seg.segmentation(image_in_resized)
        image_out_resized = image_out_resized.astype(np.uint8).squeeze()

        ## ========== semantic extraction
        # self.generate_and_publish_convex_hull(image_out_resized, msg.header.frame_id, index_care_about=2) # cross walk
        # self.generate_and_publish_convex_hull(image_out_resized, msg.header.frame_id, index_care_about=1) # road

        # NOTE: we use INTER_NEAREST because values are discrete labels
        image_out = cv2.resize(image_out_resized, (image_in.shape[1], image_in.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

        ## ========== Visualize semantic images
        # Convert network label to color
        colored_output = self.seg_color_fn(image_out, self.seg_color_ref)
        colored_output = np.squeeze(colored_output)
        colored_output = colored_output.astype(np.uint8)

        # Note: colored_ouptput is in the RGB format, if you visualize it in the opencv, you should convert it to the
        # BGR format. This is a mistake made before our publication, and by doing so we need to update all our color
        # which is a lot of work, so we don't do it.
        # colored_output = cv2.cvtColor(colored_output, cv2.COLOR_RGB2BGR)

        try:
            image_pub = self.bridge.cv2_to_imgmsg(colored_output, encoding="passthrough")
        except CvBridgeError as e:
            print(e)
        rospy.logdebug("Publish Segmented image at: %d.%09ds", msg.header.stamp.secs, msg.header.stamp.nsecs)

        image_pub.header.stamp = msg.header.stamp
        image_pub.header.frame_id = msg.header.frame_id
        if msg.header.frame_id == "camera1":
            self.image_pub_cam1.publish(image_pub)
        elif msg.header.frame_id == "camera6":
            self.image_pub_cam6.publish(image_pub)
        else:
            rospy.logwarn("publisher not spepcify for this camera frame_id %s.", msg.header.frame_id)

    def generate_and_publish_convex_hull(self, image, cam_frame_id, index_care_about=1):
        if cam_frame_id == "camera1":
            cam = self.cam1
        elif cam_frame_id == "camera6":
            cam = self.cam6

        vertice_list = generate_convex_hull(image, index_care_about=index_care_about, vis=False)

        # scale vertices to true position in original image (network output is small)
        scale_x = float(cam.imSize[0]) / image.shape[1]
        scale_y = float(cam.imSize[1]) / image.shape[0]
        for i in range(len(vertice_list)):
            vertice_list[i] = vertice_list[i] * np.array([[scale_x, scale_y]]).T

        self.cam_back_project_convex_hull(cam, vertice_list, index_care_about=index_care_about)

    def cam_back_project_convex_hull(self, cam, vertice_list, index_care_about=1):
        if len(vertice_list) == 0:
            print("vertice_list empty!")
            return

        current_time = rospy.get_rostime()
        duration = current_time - self.plane_last_update_time
        rospy.logdebug("duration: %d.%09d s", duration.secs, duration.nsecs)

        if duration.secs != 0 or duration.nsecs > 1e8:
            rospy.logwarn('too long since last update of plane %d.%09d s, please use smaller image', duration.secs,
                          duration.nsecs)

        print("vertice_list non empty!, length ", len(vertice_list))

        vertices_marker_array = MarkerArray()
        for vertices in vertice_list:
            # print(vertices)
            x = vertices
            d_vec, C_vec = cam.pixel_to_ray_vec(x)
            intersection_vec = self.plane.plane_ray_intersection_vec(d_vec, C_vec)

            self.hull_id += 1

            if index_care_about == 1:
                color = [0.8, 0., 0., 0.8]  # crosswalk is red
                vis_time = 10.0  # convex_hull marker alive time
            else:
                color = [0.0, 0, 0.8, 0.8]  # road is blue
                vis_time = 3.0
            marker = visualize_marker([0, 0, 0],
                                      mkr_id=self.hull_id,
                                      frame_id="velodyne",
                                      mkr_type="line_strip",
                                      scale=0.1,
                                      points=intersection_vec.T,
                                      lifetime=vis_time,
                                      mkr_color=color)
            vertices_marker_array.markers.append(marker)

        if index_care_about == 1:
            self.pub_crosswalk_markers.publish(vertices_marker_array)
        else:
            self.pub_road_markers.publish(vertices_marker_array)

    def plane_callback(self, msg):
        self.plane = Plane3D(msg.coef[0], msg.coef[1], msg.coef[2], msg.coef[3])
        self.plane_last_update_time = rospy.get_rostime()


def parse_args():
    """ Parse the command line arguments """
    parser = argparse.ArgumentParser(description='PycOccNet Training')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='',
        metavar='FILE',
        help='path to config file',
        type=str,
    )

    # Code inspired from https://discourse.ros.org/t/getting-python-argparse-to-work-with-a-launch-file-or-python-node/10606
    # Note that here we use sys.argv[1:-2] as the last two parameters relate to roslaunch
    args = parser.parse_args(sys.argv[1:-2])
    return args


def main():
    rospy.init_node('semantic_segmentation_node')

    cfg = get_cfg_defaults()
    args = parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    if cfg.RNG_SEED > -1:
        set_random_seed(cfg.RNG_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ss_node = SemanticSegmentationNode(cfg)
    rate = rospy.Rate(15)  # ROS Rate at 15Hz, note that from rosbag, the image comes at 12Hz

    while not rospy.is_shutdown():
        rate.sleep()


if __name__ == '__main__':
    main()
