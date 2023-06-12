#!/usr/bin/env python3
""" Semantic mapping ROS Wrapper Node

Author: Henry Zhang
Date: February 24, 2020

"""
import argparse
import cv2
import numpy as np
import os
import os.path as osp
import rospy
import sys
import hickle

# Add src directory into the path
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "../")))

from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2 as pc2
from tf import TransformListener, TransformerROS


from src.camera import camera_setup_1, camera_setup_6
from src.node_config.base_cfg import get_cfg_defaults
from src.data.confusion_matrix import ConfusionMatrix, adjust_for_mapping
from src.homography import generate_homography
from src.renderer import render_bev_map, render_bev_map_with_thresholds, apply_filter
from src.utils.utils import homogenize, dehomogenize, get_rotation_from_angle_2d
from src.utils.utils_ros import set_map_pose, get_transformation, get_transform_from_pose, create_point_cloud
from src.utils.logger import MyLogger
from src.utils.file_io import makedirs
from test.test_semantic_mapping import Test
from src.semantic_mapping import SemanticMapping


class SemanticMappingNode:
    """
    Create a semantic bird's eye view map from the LiDAR sensor and 2D semantic segmentation image. The BEV map is
    represented by a grid.

    """

    def __init__(self, cfg):
        """

        Args:
            cfg: Configuration file
        """
        # Sanity check
        assert len(cfg.LABELS) == len(cfg.LABELS_NAMES) == len(cfg.LABEL_COLORS)

        # Set up ros subscribers
        self.sub_pose = rospy.Subscriber("/current_pose", PoseStamped, self.pose_callback)
        self.image_sub_cam1 = rospy.Subscriber("/camera1/semantic", Image, self.image_callback, queue_size=1)
        self.image_sub_cam6 = rospy.Subscriber("/camera6/semantic", Image, self.image_callback, queue_size=1)

        self.depth_method = cfg.MAPPING.DEPTH_METHOD
        if self.depth_method == 'points_map':
            self.sub_pcd = rospy.Subscriber("/reduced_map", PointCloud2, self.pcd_callback)
        elif self.depth_method == 'points_raw':
            self.sub_pcd = rospy.Subscriber("/points_raw", PointCloud2, self.pcd_callback)
        else:
            rospy.logwarn("Depth estimation method set to others, use planar assumption!")

        # Set up ros publishers
        self.pub_semantic_local_map = rospy.Publisher("/semantic_local_map", Image, queue_size=5)
        self.pub_pcd = rospy.Publisher("/semantic_point_cloud", PointCloud2, queue_size=5)

        self.tf_listener = TransformListener()
        self.tf_ros = TransformerROS()
        self.bridge = CvBridge()

        # Set up the output directory
        output_dir = cfg.OUTPUT_DIR  # type:str
        if '@' in output_dir:
            # Replace @ with the project root directory
            output_dir = output_dir.replace('@', osp.join(osp.dirname(__file__), "../"))
            # Create a sub-folder in the output directory with the name of cfg.TASK_NAME
            output_dir = osp.join(output_dir, cfg.TASK_NAME)
            output_dir = osp.abspath(output_dir)

        # Set up the logger
        self.logger = MyLogger("mapping", save_dir=output_dir, use_timestamp=False)
        # Because logger will create create a sub folder "version_xxx", we need to update the output_dir
        output_dir = self.logger.save_dir
        self.output_dir = output_dir

        self.pose_queue = []
        self.cam1 = camera_setup_1()
        self.cam6 = camera_setup_6()
        rospy.logwarn("currently only for front view")

        self.pcd_frame_id = None
        self.pcd_queue = []
        self.pcd_header_queue = []

        self.save_map_to_file = False

        self.resolution = cfg.MAPPING.RESOLUTION
        self.label_names = cfg.LABELS_NAMES
        self.label_colors = np.array(cfg.LABEL_COLORS)
        self.color_remap_source = np.array(cfg.COLOR_REMAP_SOURCE)
        self.color_remap_dest = np.array(cfg.COLOR_REMAP_DEST)

        self.position_rel = np.array([[0, 0, 0]]).T
        self.yaw_rel = 0

        # This is a testing parameter, when the time stamp reach this number, the entire node will terminate.
        self.test_cut_time = cfg.TEST_END_TIME

        # load confusion matrix, we may take log probability instead
        if cfg.MAPPING.CONFUSION_MTX.LOAD_PATH != "":
            confusion_matrix = ConfusionMatrix(load_path=cfg.MAPPING.CONFUSION_MTX.LOAD_PATH)
            confusion_matrix.merge_labels(cfg.SRC_INDICES, cfg.DST_INDICES)
            self.confusion_matrix = confusion_matrix.get_submatrix(cfg.LABELS, to_probability=True, use_log=False)
            self.confusion_matrix = adjust_for_mapping(self.confusion_matrix, factor=cfg.MAPPING.REWEIGHT_FACTOR)
            self.confusion_matrix = np.log(self.confusion_matrix)
            print('confusion_matrix:', self.confusion_matrix)
        else:
            # Use Identity confusion matrix
            self.confusion_matrix = np.eye(len(self.label_names))

        # Print the configuration to user
        self.logger.log("Running with configuration:\n" + str(cfg))

        self.ground_truth_dir = cfg.GROUND_TRUTH_DIR
        self.input_list = []
        self.unique_input_dict = {}
        self.input_dir = cfg.MAPPING.INPUT_DIR
        self.mapper = SemanticMapping(cfg, self.logger)


    def pcd_callback(self, msg):
        """ Callback function for the point cloud data.
        Store data to queue for synchronization. """
        rospy.logdebug("pcd data frame_id %s", msg.header.frame_id)
        rospy.logdebug("pcd data received")
        rospy.logdebug("pcd size: %d, %d", msg.height, msg.width)
        if len(self.pcd_queue) > 10:
            rospy.logwarn("pcd queue size: %d", len(self.pcd_queue))
        pcd = np.empty((4, msg.width))
        for i, el in enumerate(pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)):
            pcd[:, i] = el
        self.pcd_queue.append(pcd)
        self.pcd_header_queue.append(msg.header)
        self.pcd_frame_id = msg.header.frame_id


    def update_pcd(self, target_stamp):
        """
        Find the closest point cloud wrt the target_stamp

        We first want to find the smallest stamp that is larger than the timestamp, then compare it with the
        largest stamp that is smaller than the timestamp. and then pick the smallest one as the result. If such
        condition does not exist, i.e. all the stamps are smaller than the time stamp, we just pick the latest one.

        Args:
            target_stamp:

        Returns:

        """
        for i in range(len(self.pcd_header_queue) - 1):
            if self.pcd_header_queue[i + 1].stamp > target_stamp:
                if self.pcd_header_queue[i].stamp < target_stamp:
                    diff_2 = self.pcd_header_queue[i + 1].stamp - target_stamp
                    diff_1 = target_stamp - self.pcd_header_queue[i].stamp
                    if diff_1 > diff_2:
                        header = self.pcd_header_queue[i + 1]
                        pcd = self.pcd_queue[i + 1]
                    else:
                        header = self.pcd_header_queue[i]
                        pcd = self.pcd_queue[i]
                    self.pcd_header_queue = self.pcd_header_queue[i::]
                    self.pcd_queue = self.pcd_queue[i::]
                    rospy.logdebug("Setting current pcd at: %d.%09ds", header.stamp.secs, header.stamp.nsecs)
                    return pcd, header.stamp
        header = self.pcd_header_queue[-1]
        pcd = self.pcd_queue[-1]
        self.pcd_header_queue = self.pcd_header_queue[-1::]
        self.pcd_queue = self.pcd_queue[-1::]
        rospy.logdebug("Setting current pcd at: %d.%09ds", header.stamp.secs, header.stamp.nsecs)
        return pcd, header.stamp


    def pose_callback(self, msg):
        """ Callback function for the pose data.
        Store data to queue for synchronization. """
        rospy.logdebug("Getting pose at: %d.%09ds", msg.header.stamp.secs, msg.header.stamp.nsecs)
        self.pose_queue.append(msg)
        rospy.logdebug("Pose queue length: %d", len(self.pose_queue))


    def set_global_map_pose(self):
        """ Send /global_map pose to TF

        Global map origin is shifted to the min x, y point in the point map
        so that the entire map will have positive values """
        pose = Pose()
        pose.position.x = abs(self.mapper.map_boundary[0][0])  # min x
        pose.position.y = abs(self.mapper.map_boundary[1][0])  # min y
        pose.position.z = 0.0
        pose.orientation.w = 1.0
        set_map_pose(pose, '/world', 'global_map')


    def update_pose(self, target_stamp):
        """
        Find the closest pose wrt the target_stamp.

        This is the same implementation as the update_pcd().
        """
        for i in range(len(self.pose_queue) - 1):
            if self.pose_queue[i + 1].header.stamp > target_stamp:
                if self.pose_queue[i].header.stamp < target_stamp:
                    diff_2 = self.pose_queue[i + 1].header.stamp - target_stamp
                    diff_1 = target_stamp - self.pose_queue[i].header.stamp
                    if diff_1 > diff_2:
                        msg = self.pose_queue[i + 1]
                    else:
                        msg = self.pose_queue[i]
                    self.pose_queue = self.pose_queue[i::]
                    rospy.logdebug("Setting current pose at: %d.%09ds", msg.header.stamp.secs, msg.header.stamp.nsecs)
                    return msg.pose, msg.header.stamp
        msg = self.pose_queue[-1]
        self.pose_queue = self.pose_queue[-1::]
        rospy.logdebug("Setting current pose at: %d.%09ds", msg.header.stamp.secs, msg.header.stamp.nsecs)
        return msg.pose, msg.header.stamp


    def image_callback(self, msg):
        """
        The callback function for the camera image. When the semantic camera image is published, this function will be
        invoked and generate a BEV semantic map from the image.
        """
        if msg.header.stamp.secs >= self.test_cut_time:
            self.save_map_to_file = True
        else:
            rospy.loginfo('{} seconds to end time.'.format(int(self.test_cut_time - msg.header.stamp.secs)))
        self.logger.log("Mapping {} image at: {}s".format(msg.header.frame_id, msg.header.stamp.to_sec()))
        try:
            image_in = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)
            return

        if msg.header.frame_id == "camera1":
            projection_matrix = self.cam1.P
        elif msg.header.frame_id == "camera6":
            projection_matrix = self.cam6.P
        else:
            rospy.logwarn("cannot find camera for frame_id %s", msg.header.frame_id)

        if self.depth_method == 'points_map' or self.depth_method == 'points_raw':
            if len(self.pcd_header_queue) == 0:
                return
            pcd, _ = self.update_pcd(msg.header.stamp)

        if len(self.pose_queue) == 0: return
        pose, pose_time = self.update_pose(msg.header.stamp)
        self.set_global_map_pose()

        T_base_to_origin = get_transform_from_pose(pose)

        T_local_to_base, _, _, _ = get_transformation(frame_from='/global_map', time_from=rospy.Time(0),
                                                      frame_to='/base_link', time_to=pose_time,
                                                      static_frame='world',
                                                      tf_listener=self.tf_listener, tf_ros=self.tf_ros)
        
        map, pcd_in_range, pcd_label = self.mapper.mapping(
            pcd, 
            self.pcd_frame_id, 
            image_in, 
            projection_matrix, 
            T_base_to_origin, 
            T_local_to_base)

        rospy.logdebug("Finished Mapping image at: %d.%09ds", msg.header.stamp.secs, msg.header.stamp.nsecs)

        debug = False
        if self.save_map_to_file or debug:
            print('Done update mapping')
            color_map = self.mapper.get_color_map(map)
            print('Done redering map.')
            
            # Publish the image
            try:
                image_pub = self.bridge.cv2_to_imgmsg(color_map, encoding="passthrough")
                self.pub_semantic_local_map.publish(image_pub)
            except CvBridgeError as e:
                print(e) 
            
            # raw map can be quite large, 1-2 GB
            output_dir = self.output_dir
            makedirs(output_dir, exist_ok=True)
            # np.save(osp.join(output_dir, "raw_map.npy"), map)

            output_file = osp.join(output_dir, "global_map_hrnet.png")
            print("Saving image to: ", output_file)
            cv2.imwrite(output_file, color_map)

            # evaluate
            if self.ground_truth_dir != "":
                test = Test(ground_truth_dir=self.ground_truth_dir, logger=self.logger)
                test.test_single_map(color_map)


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
    rospy.init_node('semantic_mapping')

    cfg = get_cfg_defaults()
    args = parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    sm = SemanticMapping(cfg)
    rospy.spin()


if __name__ == "__main__":
    main()
