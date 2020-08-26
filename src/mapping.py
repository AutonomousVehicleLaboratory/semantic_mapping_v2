#!/usr/bin/env python
""" Semantic mapping

Author: Henry Zhang
Date:February 24, 2020

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
from tf.transformations import euler_matrix

from src.camera import camera_setup_1, camera_setup_6
from src.config.base_cfg import get_cfg_defaults
from src.data.confusion_matrix import ConfusionMatrix
from src.homography import generate_homography
from src.renderer import render_bev_map, render_bev_map_with_thresholds, apply_filter
from src.utils.utils import homogenize, dehomogenize, get_rotation_from_angle_2d
from src.utils.utils_ros import set_map_pose, get_transformation, get_transform_from_pose, create_point_cloud
from src.utils.logger import MyLogger
from src.utils.file_io import makedirs
from test.test_semantic_mapping import Test


class SemanticMapping:
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
        self.image_sub_cam1 = rospy.Subscriber("/camera1/semantic", Image, self.image_callback)
        self.image_sub_cam6 = rospy.Subscriber("/camera6/semantic", Image, self.image_callback)

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

        self.pose = None
        self.pose_queue = []
        self.pose_time = None
        self.cam1 = camera_setup_1()
        self.cam6 = camera_setup_6()
        rospy.logwarn("currently only for front view")

        self.pcd = None
        self.pcd_frame_id = None
        self.pcd_queue = []
        self.pcd_header_queue = []
        self.pcd_time = None
        self.pcd_range_max = cfg.MAPPING.PCD.RANGE_MAX
        self.use_pcd_intensity = cfg.MAPPING.PCD.USE_INTENSITY

        # The coordinate of the map is defined as map[x, y]
        self.map = None
        self.map_pose = None
        self.save_map_to_file = False
        self.map_boundary = cfg.MAPPING.BOUNDARY
        self.resolution = cfg.MAPPING.RESOLUTION
        self.label_names = cfg.LABELS_NAMES
        self.label_colors = np.array(cfg.LABEL_COLORS)

        self.map_height = int((self.map_boundary[0][1] - self.map_boundary[0][0]) / self.resolution)
        self.map_width = int((self.map_boundary[1][1] - self.map_boundary[1][0]) / self.resolution)
        self.map_depth = len(self.label_names)

        self.position_rel = np.array([[0, 0, 0]]).T
        self.yaw_rel = 0

        self.preprocessing()

        # This is a testing parameter, when the time stamp reach this number, the entire node will terminate.
        self.test_cut_time = cfg.TEST_END_TIME

        if cfg.MAPPING.CONFUSION_MTX.LOAD_PATH != "":
            confusion_matrix = ConfusionMatrix(load_path=cfg.MAPPING.CONFUSION_MTX.LOAD_PATH)
            self.confusion_matrix = confusion_matrix.get_submatrix(cfg.LABELS, to_probability=True, use_log=True)
        else:
            # Use Identity confusion matrix
            self.confusion_matrix = np.eye(len(self.label_names))

        # Print the configuration to user
        self.logger.log("Running with configuration:\n" + str(cfg))

        self.ground_truth_dir = cfg.GROUND_TRUTH_DIR
        self.input_list = []
        self.unique_input_dict = {}
        self.input_dir = cfg.MAPPING.INPUT_DIR

    def preprocessing(self):
        """ Setup constant matrices """
        self.T_velodyne_to_basklink = self.set_velodyne_to_baselink()
        self.T_cam1_to_base = np.matmul(self.T_velodyne_to_basklink, self.cam1.T)
        self.T_cam6_to_base = np.matmul(self.T_velodyne_to_basklink, self.cam6.T)

        self.discretize_matrix_inv = np.array([
            [self.resolution, 0, self.map_boundary[0][0]],
            [0, self.resolution, self.map_boundary[1][1]],
            [0, 0, 1],
        ]).astype(np.float)
        self.discretize_matrix = np.linalg.inv(self.discretize_matrix_inv)

        self.anchor_points = np.array([
            [self.map_width, self.map_width / 3, self.map_width, self.map_width / 3],
            [self.map_height / 4, self.map_height / 4, self.map_height * 3 / 4, self.map_height * 3 / 4],
        ])

        self.anchor_points_2 = np.array([
            [self.map_width, self.map_width / 2, self.map_width / 2, self.map_width],
            [self.map_height / 4, self.map_height / 4, self.map_height * 3 / 4, self.map_height * 3 / 4],
        ])

    def set_velodyne_to_baselink(self):
        rospy.logwarn("velodyne to baselink from TF is tunned, current version fits best.")
        T = euler_matrix(0., 0.140, 0.)
        t = np.array([[2.64, 0, 1.98]]).T
        T[0:3, -1::] = t
        return T

    def pcd_callback(self, msg):
        """ Callback function for the point cloud dataset """
        rospy.logwarn("pcd data frame_id %s", msg.header.frame_id)
        rospy.logdebug("pcd data received")
        rospy.logdebug("pcd size: %d, %d", msg.height, msg.width)
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
        rospy.logdebug("Getting pose at: %d.%09ds", msg.header.stamp.secs, msg.header.stamp.nsecs)
        self.pose_queue.append(msg)
        if msg.header.stamp.secs >= self.test_cut_time:
            self.save_map_to_file = True
        rospy.logdebug("Pose queue length: %d", len(self.pose_queue))

    def set_global_map_pose(self):
        """ global map origin is shifted to the min x, y point in the point map
        so that the entire map will have positive values """
        pose = Pose()
        pose.position.x = -1369.0496826171875  # min x
        pose.position.y = -562.84814453125  # min y
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
        self.logger.log("Mapping image at: {}.{:.9f}s".format(msg.header.stamp.secs, msg.header.stamp.nsecs))
        try:
            image_in = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)
            return

        if msg.header.frame_id == "camera1":
            camera_calibration = self.cam1
        elif msg.header.frame_id == "camera6":
            camera_calibration = self.cam6
        else:
            rospy.logwarn("cannot find camera for frame_id %s", msg.header.frame_id)

        if self.depth_method in ['points_map', 'points_raw']:
            if len(self.pcd_header_queue) == 0: return
            self.pcd, self.pcd_time = self.update_pcd(msg.header.stamp)

        if len(self.pose_queue) == 0: return
        self.pose, self.pose_time = self.update_pose(msg.header.stamp)
        self.set_global_map_pose()

        self.mapping(image_in, self.pose, camera_calibration)

        rospy.logdebug("Finished Mapping image at: %d.%09ds", msg.header.stamp.secs, msg.header.stamp.nsecs)

    def mapping(self, semantic_image, pose, camera_calibration):
        """
        Receives the semantic segmentation image, the pose of the vehicle, and the calibration of the camera,
        we will build a semantic point cloud, then project it into the 2D bird's eye view coordinates.

        Args:
            semantic_image: 2D semantic image
            pose: vehicle pose
            camera_calibration: The calibration information of the camera
        """
        # Initialize the map
        if self.map is None:
            self.map = np.zeros((self.map_height, self.map_width, self.map_depth))
            # self.unique_input_dict["initial_map"] = np.array(self.map)
            # self.unique_input_dict["camera_calibration"] = camera_calibration

        if self.depth_method in ['points_map', 'points_raw']:
            frame_input_dict = {"pcd": np.array(self.pcd),
                                "pcd_frame_id": self.pcd_frame_id,
                                "semantic_image": np.array(semantic_image),
                                "pose": pose}
            self.input_list.append(frame_input_dict)
            pcd_in_range, pcd_label = self.project_pcd(self.pcd, self.pcd_frame_id, semantic_image, pose,
                                                       camera_calibration)
            pcd_pub = create_point_cloud(pcd_in_range[0:3].T, pcd_label.T, frame_id=self.pcd_frame_id)
            self.pub_pcd.publish(pcd_pub)

            self.map = self.update_map(self.map, pcd_in_range, pcd_label)
        else:
            self.map = self.update_map_planar(self.map, semantic_image, camera_calibration)

        if self.save_map_to_file:
            # with open(os.path.join(self.input_dir, "input_list.hkl"), 'wb') as fp:
            #     print("writing input_list ...")
            #     hickle.dump(self.input_list, fp, mode='w')    
                        
            output_dir = self.output_dir
            makedirs(output_dir, exist_ok=True)
            # np.save(osp.join(output_dir, "map.npy"), self.map)

            self.map = apply_filter(self.map)  # smooth the labels to fill black holes

            color_map = render_bev_map(self.map, self.label_colors)
            # color_map = render_bev_map_with_thresholds(self.map, self.label_colors, priority=[3, 4, 0, 2, 1],
            #                                            thresholds=[0.1, 0.1, 0.5, 0.20, 0.05])

            output_file = osp.join(output_dir, "global_map.png")
            print("Saving image to", output_file)
            cv2.imwrite(output_file, color_map)

            # evaluate
            if self.ground_truth_dir != "":
                test = Test(ground_truth_dir=self.ground_truth_dir, logger=self.logger)
                test.test_single_map(color_map)

            # Publish the image
            try:
                image_pub = self.bridge.cv2_to_imgmsg(color_map, encoding="passthrough")
                self.pub_semantic_local_map.publish(image_pub)
            except CvBridgeError as e:
                print(e)

            # TODO: This line of code is just for debugging purpose
            rospy.signal_shutdown('Done with the mapping')

    def project_pcd(self, pcd, pcd_frame_id, image, pose, camera_calibration):
        """
        Extract labels of each point in the pcd from image
        Args:
            camera_calibration:camera calibration information, it includes the camera projection matrix.

        Returns: Point cloud that are visible in the image, and their associated labels

        """
        if pcd is None: return
        if pcd_frame_id != "velodyne":
            T_base_to_origin = get_transform_from_pose(pose)
            T_origin_to_velodyne = np.linalg.inv(np.matmul(T_base_to_origin, self.T_velodyne_to_basklink))

            pcd_velodyne = np.matmul(T_origin_to_velodyne, homogenize(pcd[0:3, :]))
        else:
            pcd_velodyne = homogenize(pcd[0:3, :])

        IXY = dehomogenize(np.matmul(camera_calibration.P, pcd_velodyne)).astype(np.int32)

        # Only use the points in the front.
        mask_positive = np.logical_and(0 < pcd_velodyne[0, :], pcd_velodyne[0, :] < self.pcd_range_max)

        # Only select the points that project to the image
        mask = np.logical_and(np.logical_and(0 <= IXY[0, :], IXY[0, :] < image.shape[1]),
                              np.logical_and(0 <= IXY[1, :], IXY[1, :] < image.shape[0]))
        mask = np.logical_and(mask, mask_positive)

        masked_pcd = pcd[:, mask]
        image_idx = IXY[:, mask]
        label = image[image_idx[1, :], image_idx[0, :]].T

        return masked_pcd, label

    def update_map(self, map, pcd, label):
        """
        Project the semantic point cloud on the BEV map

        Args:
            map: np.ndarray with shape (H, W, C). H is the height, W is the width, and C is the semantic class.
            pcd: np.ndarray with shape (4, N). N is the number of points. The point cloud
            label: np.ndarray with shape (3, N). N is the number of points. The RGB label of each point cloud.

        Returns:
            Updated map
        """
        normal = np.array([[0.0, 0.0, 1.0]]).T  # The normal of the z axis
        pcd_origin_offset = np.array([[1369.0496826171875], [562.84814453125], [0.0]]) # pcd origin with respect to map origin
        pcd_local = pcd[0:3] + pcd_origin_offset
        pcd_on_map = pcd_local - np.matmul(normal, np.matmul(normal.T, pcd_local))
        # Discretize point cloud into grid, Note that here we are basically doing the nearest neighbor search
        pcd_pixel = ((pcd_on_map[0:2, :] - np.array([[self.map_boundary[0][0]], [self.map_boundary[1][0]]]))
                     / self.resolution).astype(np.int32)
        on_grid_mask = np.logical_and(np.logical_and(0 <= pcd_pixel[0, :], pcd_pixel[0, :] < self.map_height),
                                      np.logical_and(0 <= pcd_pixel[1, :], pcd_pixel[1, :] < self.map_width))

        # Update corresponding labels
        for i, label_name in enumerate(self.label_names):
            # Code explanation:
            # We first do an elementwise comparison
            # a = (label == self.label_colors[i].reshape(3, 1))
            # Then we do a logical AND among the rows of a, represented by *a.
            idx = np.logical_and(*(label == self.label_colors[i].reshape(3, 1)))
            idx_mask = np.logical_and(idx, on_grid_mask)

            # Update the local map with Bayes update rule
            # map[pcd_pixel[0, idx_mask], pcd_pixel[1, idx_mask], :] has shape (n, num_classes)
            map[pcd_pixel[0, idx_mask], pcd_pixel[1, idx_mask], :] += self.confusion_matrix[:, i].reshape(1, -1)

            # LiDAR intensity augmentation
            if not self.use_pcd_intensity: continue

            # For all the points that have been classified as land, we augment its count by looking at its intensity
            # print(label_name)
            if label_name == "lane":
                intensity_mask = np.logical_or(pcd[3] < 2, pcd[3] > 14)  # These thresholds are found by experiment.
                intensity_mask = np.logical_and(intensity_mask, idx_mask)

                # 2 is an experimental number which we think is good enough to connect the lane on the side.
                # Too large the lane will be widen, too small the lane will be fragmented.
                map[pcd_pixel[0, intensity_mask], pcd_pixel[1, intensity_mask], i] += 2

                # For the region where there is no intensity by our network detected as lane, we will degrade its
                # threshold
                # non_intensity_mask = np.logical_and(~intensity_mask, idx_mask)
                # map[pcd_pixel[1, non_intensity_mask], pcd_pixel[0, non_intensity_mask], i] -= 0.5

        return map

    def update_map_planar(self, map_local, image, cam):
        """ Project the semantic image onto the map plane and update it """

        # points_map = homogenize(np.array(self.anchor_points_2))
        # points_local = np.matmul(self.discretize_matrix_inv, points_map)
        # points_local[2, :] = 0
        # points_local = homogenize(points_local)

        T_local_to_base, _, _, _ = get_transformation(frame_from='/global_map', time_from=rospy.Time(0),
                                                      frame_to='/base_link', time_to=self.pose_time,
                                                      static_frame='world',
                                                      tf_listener=self.tf_listener, tf_ros=self.tf_ros)
        T_base_to_velodyne = np.linalg.inv(self.T_velodyne_to_basklink)
        T_local_to_velodyne = np.matmul(T_base_to_velodyne, T_local_to_base)

        points_base_link = np.array([[30, 10, 10, 30],
                                    [0, 2, 5, 15],
                                    [0, 0, 0, 0]], dtype=np.float)
        points_global = np.matmul(np.linalg.inv(T_local_to_base), homogenize(points_base_link))
        anchor_points_global = ((points_global[0:2, :] - np.array([[self.map_boundary[0][0]], [self.map_boundary[1][0]]]))
                     / self.resolution).astype(np.int32)

        # compute new points
        points_velodyne = np.matmul(T_local_to_velodyne, points_global)
        points_image = dehomogenize(np.matmul(cam.P, points_velodyne)).astype(np.int)

        # generate homography
        image_on_map = generate_homography(image, points_image.T, anchor_points_global.T, vis=True,
                                           out_size=[self.map_height, self.map_width])
        sep = int((8 - self.map_boundary[0][0]) / self.resolution)
        mask = np.ones(map_local.shape[0:2])
        # mask[:, 0:sep] = 0
        idx_mask_3 = np.zeros([map_local.shape[0], map_local.shape[1], 3])

        for i in range(len(self.label_names)):
            idx = image_on_map[:, :, 0] == self.label_names[i]
            idx_mask = np.logical_and(idx, mask)
            map_local[idx_mask, i] += 1
            # idx_mask_3[idx_mask] = self.catogories_color[i]
        # cv2.imshow("mask", idx_mask_3.astype(np.uint8))
        # cv2.waitKey(1)

        map_local[map_local < 0] = 0

        # threshold and normalize
        # map_local[map_local > self.map_value_max] = self.map_value_max
        # print("max:", np.max(map_local))
        # normalized_map = self.normalize_map(map_local)

        return map_local

    def add_car_to_map(self, color_map):
        """
        Warning: This function is not tested, may have bug!
        Args:
            color_map:

        Returns:

        """

        """ visualize ego car on the color map """
        # setting parameters
        length = 4.0
        width = 1.8
        mask_length = int(length / self.resolution)
        mask_width = int(width / self.resolution)
        car_center = np.array([[length / 4, width / 2]]).T / self.resolution
        discretize_matrix_inv = np.array([
            [self.resolution, 0, -length / 4],
            [0, -self.resolution, width / 2],  # Warning: double check the sign of -self.resolution
            [0, 0, 1]
        ])

        # pixels in ego frame
        Ix = np.tile(np.arange(0, mask_length), mask_width)
        Iy = np.repeat(np.arange(0, mask_width), mask_length)
        Ixy = np.vstack([Ix, Iy])

        # transform to map frame
        R = get_rotation_from_angle_2d(self.yaw_rel)
        Ixy_map = np.matmul(R, Ixy - car_center) + self.position_rel[0:2].reshape([2, 1]) / self.resolution + \
                  np.array([[-self.map_boundary[0][0] / self.resolution, self.map_height / 2]]).T
        Ixy_map = Ixy_map.astype(np.int)

        # setting color
        color_map[Ixy_map[1, :], Ixy_map[0, :], :] = [255, 0, 0]
        return color_map

    def get_extrinsics(self, pose, camera_id):
        T_base_to_origin = get_transform_from_pose(pose)

        # from camera to origin
        if camera_id == "camera1":
            T_cam_to_origin = np.matmul(T_base_to_origin, self.T_cam1_to_base)
        elif camera_id == "camera6":
            T_cam_to_origin = np.matmul(T_base_to_origin, self.T_cam6_to_base)
        else:
            rospy.logwarn("unable to find camera to base for camera_id %s", camera_id)
        T_origin_to_cam = np.linalg.inv(T_cam_to_origin)
        extrinsics = T_origin_to_cam[0:3]
        return extrinsics


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
