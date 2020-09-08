#!/usr/bin/env python
""" Semantic mapping

Author: Henry Zhang
Date:February 24, 2020

"""
import argparse
import cv2
import numpy as np
import os.path as osp
import rospy
import sys

# Add src directory into the path
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "../")))

# Package that cannot be installed by pip (or is outdated because of the specific ROS version we use)
try:
    from cv_bridge import CvBridge, CvBridgeError
    from geometry_msgs.msg import PoseStamped, Pose
    from sensor_msgs import point_cloud2
    from sensor_msgs.msg import Image, PointCloud2
    from tf import TransformListener, TransformerROS
    from tf.transformations import euler_matrix
except:
    pass

from collections import deque

from src.camera import build_camera_model
from src.config.mapping import get_cfg_defaults
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

        # Set up ROS subscribers
        self._subscribers = {}
        self._subscribers["pose"] = rospy.Subscriber("/current_pose", PoseStamped, self._pose_callback)

        self._target_cameras = cfg.SEM_SEG.TARGET_CAMERAS
        for cam in self._target_cameras:
            self._subscribers[cam] = rospy.Subscriber("/{}/semantic".format(cam), Image, self._image_callback)

        # The possible point cloud source
        # dense_pcd - dense point cloud we generated from the map reduce function.
        # raw_pcd - the point cloud returned directly from Velodyne sensor.
        self._point_cloud_source = cfg.MAPPING.POINT_CLOUD_SOURCE
        if self._point_cloud_source == "dense_pcd":
            self._subscribers["pcd"] = rospy.Subscriber("/reduced_map", PointCloud2, self._pcd_callback)
        elif self._point_cloud_source == "raw_pcd":
            self._subscribers["pcd"] = rospy.Subscriber("/points_raw", PointCloud2, self._pcd_callback)
        else:
            raise NotImplementedError

        # Set up ROS publishers
        self._publishers = {}
        self._publishers["semantic_local_map"] = rospy.Publisher("/semantic_local_map", Image, queue_size=5)
        self._publishers["pcd"] = rospy.Publisher("/semantic_point_cloud", PointCloud2, queue_size=5)

        # Launch ROS specific classes
        self._tf_listener = TransformListener()
        self._tf_ros = TransformerROS()
        self._bridge = CvBridge()

        # Set up the output directory
        output_dir = cfg.OUTPUT_DIR  # type:str
        # If output directory is empty, we set it to the [project root directory]/outputs/[TASK_NAME]
        if not output_dir:
            output_dir = osp.abspath(osp.join(osp.dirname(__file__), "../outputs"))
        output_dir = osp.join(output_dir, cfg.TASK_NAME)

        # Set up the logger
        self.logger = MyLogger("mapping", save_dir=output_dir, use_timestamp=True)
        # Because logger will create create a sub folder "version_xxx", we need to update the output_dir
        output_dir = self.logger.save_dir
        self.output_dir = output_dir

        # Load camera model
        self.camera_models = {}
        for cam in self._target_cameras:
            self.camera_models[cam] = build_camera_model(cam)

        # ROS data queue
        # Each element in the queue should be a tuple of (header, data), i.e. the same format that
        # self._find_closest_data uses.
        self._pose_queue = deque()
        self._pcd_queue = deque()

        #
        self._pcd_range_max = cfg.MAPPING.PCD.RANGE_MAX  # The maximum range of the point cloud
        self._use_pcd_intensity = cfg.MAPPING.PCD.USE_INTENSITY

        self._map = None  # The coordinate of the map is defined as map[x, y]
        self._map_height = int((self.map_boundary[0][1] - self.map_boundary[0][0]) / self.resolution)
        self._map_width = int((self.map_boundary[1][1] - self.map_boundary[1][0]) / self.resolution)
        self._label_names = cfg.LABELS_NAMES
        self._map_depth = len(self.label_names)

        self._load_constant()
        self._set_global_map_pose()

    def _load_constant(self):
        """ Load the constant  """

        # Determine the transformation from velodyne to baselink
        # Note that the number here is tuned by experiments. Current version fits the best.
        T = euler_matrix(0., 0.140, 0.)
        t = np.array([[2.64, 0, 1.98]]).T
        T[0:3, -1::] = t
        self.T_velodyne_to_basklink = T

        # Determine the transformation from camera to baselink
        self.T_camera_to_baselink = {}
        for cam in self._target_cameras:
            self.T_camera_to_baselink = np.matmul(self.T_velodyne_to_basklink, self.camera_models[cam].T)

    def _set_global_map_pose(self):
        """
        Set the origin of the pose in the global frame to the min x, y point in the point map so that the entire map
        will have positive values
        """
        pose = Pose()
        # Number provided here is hard coded.
        pose.position.x = -1369.0496826171875  # min x
        pose.position.y = -562.84814453125  # min y
        pose.position.z = 0.0
        pose.orientation.w = 1.0
        set_map_pose(pose, '/world', 'global_map')

    def _find_closest_data(self, queue, target_stamp):
        """
        Find the closest data B in queue w.r.t the target_stamp. Note that all the data that happens earlier than B,
        including B itself will be removed from the queue. By doing so we can avoid re-association of previous data.
        We assume that the time in queue is monotonically increase.

        We first want to find the smallest stamp that is larger than the timestamp, then compare it with the
        largest stamp that is smaller than the timestamp. and then pick the smallest one as the result. If such
        condition does not exist, i.e. all the stamps are smaller than the time stamp, we just pick the latest one.

        Args:
            queue (deque): Each element in queue should be a tuple of (header, data) where we don't care the data
                format. The header should have a header.stamp which contains its corresponding time stamp.
            target_stamp (float): The target time stamp

        Returns:
            if the queue is empty, we return None.

        """
        if len(queue) == 0:
            return None

        # Identify the closest time stamp that is smaller than target_stamp
        prev_elem = None
        selected_elem = None
        while len(queue) > 0:
            curr_elem = queue.popleft()
            curr_stamp = curr_elem[0].stamp
            if curr_stamp > target_stamp:
                if prev_elem is None:
                    selected_elem = curr_elem
                    break
                else:
                    prev_stamp = prev_elem[0].stamp
                    diff_1 = np.abs(curr_stamp - target_stamp)
                    diff_2 = np.abs(prev_stamp - target_stamp)
                    if diff_1 < diff_2:
                        # Pick current element
                        selected_elem = curr_elem
                    else:
                        # Pick previous elem and put current element back
                        selected_elem = prev_elem
                        queue.appendleft(curr_elem)
                    break

            prev_elem = curr_elem

        # If no element that are greater than target stamp, return the lastest one
        if selected_elem is None:
            return prev_elem
        else:
            return selected_elem

    def _pcd_callback(self, msg):
        """
        The callback function for the incoming point cloud data

        Args:
            msg (PointCloud2): refer to this http://docs.ros.org/melodic/api/sensor_msgs/html/msg/PointCloud2.html

        """
        # if False:
        #     self.logger.warning("pcd data frame_id %s", msg.header.frame_id)
        #     self.logger.debug("pcd data received")
        #     self.logger.debug("pcd size: %d, %d", msg.height, msg.width)
        #     self.logger.warning("pcd queue size: %d", len(self.pcd_queue))
        pcd = np.empty((4, msg.width))
        for i, el in enumerate(point_cloud2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)):
            pcd[:, i] = el
        self.pcd_queue.append((msg.header, pcd))

    def _pose_callback(self, msg):
        """
        The callback function for the incoming pose data

        Args:
            msg (PoseStamped): http://docs.ros.org/melodic/api/geometry_msgs/html/msg/PoseStamped.html

        """
        self.pose_queue.append((msg.header, msg))

        # For testing purpose, if the receiving pose is greater than the test_cut_time, we stop the test and save the
        # result.
        if msg.header.stamp.secs >= self.test_cut_time:
            self.save_map_to_file = True

    def _image_callback(self, msg):
        """
        The callback function for the incoming camera image. When a semantic camera image is published, this function
        will be invoked and generate a BEV semantic occupancy grid from the image.

        Args:
            msg (Image):

        """
        try:
            image_in = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)
            return

        if msg.header.frame_id in self._target_cameras:
            camera_model = self.camera_models[msg.header.frame_id]
        else:
            self.logger.log("Unknown camera model {}".format(msg.header.frame_id))
            return

        # Get the associated point cloud and point cloud for the image
        retval = self._find_closest_data(self._pcd_queue, msg.header.stamp)
        if retval is None: return  # If retval is None, we don't have associated point cloud, cannot proceed.
        pcd_header, pcd = retval

        retval = self._find_closest_data(self._pose_queue, msg.header.stamp)
        if retval is None: return
        pose_header, pose = retval

        self.mapping(image_in, pose, pcd, camera_model)

    def mapping(self, semantic_image, pose, pcd, camera_model):
        """
        Given a semantic image, its associated vehicle's pose of the vehicle, LiDAR point cloud and the camera model,
        this function will build a semantic point cloud, then project it into the 2D bird's eye view coordinates.

        Args:
            semantic_image (np.ndarray): 2D semantic image
            pose (PoseStamped): Vehicle pose in the world frame
            pcd (np.ndarray): Point cloud
            camera_model: Contains the calibration information of the camera
        """
        # Initialize the map
        if self._map is None:
            self._map = np.zeros((self._map_height, self._map_width, self._map_depth))

        pcd_in_range, pcd_label = self.project_pcd(pcd, self._point_cloud_source, semantic_image, pose, camera_model)

        # Maybe I should return this function and let its parent to handle this.
        # Publish point cloud message
        # pcd_msg = create_point_cloud(pcd_in_range[:3].T, pcd_label.T, frame_id=???)
        # self._publishers["pcd"].publish(pcd_msg)

        self._map = self.update_map(self._map, pcd_in_range, pcd_label)

        return

        if self.depth_method in ['points_map', 'points_raw']:
            frame_input_dict = {"pcd": np.array(self.pcd),
                                "pcd_frame_id": self.pcd_frame_id,
                                "semantic_image": np.array(semantic_image),
                                "pose": pose}
            self.input_list.append(frame_input_dict)
            pcd_in_range, pcd_label = self.project_pcd(self.pcd, self.pcd_frame_id, semantic_image, pose,
                                                       camera_model)
            pcd_pub = create_point_cloud(pcd_in_range[0:3].T, pcd_label.T, frame_id=self.pcd_frame_id)
            self.pub_pcd.publish(pcd_pub)

            self.map = self.update_map(self.map, pcd_in_range, pcd_label)
        else:
            self.map = self.update_map_planar(self.map, semantic_image, camera_model)

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
                image_pub = self._bridge.cv2_to_imgmsg(color_map, encoding="passthrough")
                self.pub_semantic_local_map.publish(image_pub)
            except CvBridgeError as e:
                print(e)

            # TODO: This line of code is just for debugging purpose
            rospy.signal_shutdown('Done with the mapping')

    def project_pcd(self, pcd, pcd_source, image, pose, camera_model):
        """
        Project point cloud into the image and find the associate pixel value for each point.

        Args:
            pcd (np.ndarray): Point cloud. Depending on the source of the point cloud, If it is from dense point
                cloud, then it is in the world frame. If it is from raw point cloud, then it is in the sensor's frame.
            pcd_source (str): The source of the point cloud, please refer to self._point_cloud_source.
            image (np.ndarray): 2D image, the shape of the image is (h, w, c)
            pose (PoseStamped): Vehicle pose in the world frame
            camera_model: The camera model, it contains the camera calibration information.

        Returns:
            masked_pcd (np.ndarray): Points that are visible in the image, in the same frame as the input pcd.
                shape = (4, n)
            label (np.ndarray): the pixel value for these visible points. shape = (3, n)

        """
        if pcd is None: return
        if pcd_source == "raw_pcd":
            pcd_in_lidar_frame = homogenize(pcd[0:3, :])
        elif pcd_source == "dense_pcd":
            # Transform the point cloud from world frame to the LiDAR sensor's frame
            T_base_to_origin = get_transform_from_pose(pose)
            T_origin_to_velodyne = np.linalg.inv(np.matmul(T_base_to_origin, self.T_velodyne_to_basklink))
            pcd_in_lidar_frame = np.matmul(T_origin_to_velodyne, homogenize(pcd[0:3, :]))
        else:
            raise NotImplementedError

        projected_pts = dehomogenize(np.matmul(camera_model.P, pcd_in_lidar_frame)).astype(np.int32)

        # Only use the points in the front.
        mask_positive = np.logical_and(0 < pcd_in_lidar_frame[0, :], pcd_in_lidar_frame[0, :] < self._pcd_range_max)

        # Only select the points that project to the image
        mask = np.logical_and(np.logical_and(0 <= projected_pts[0, :], projected_pts[0, :] < image.shape[1]),
                              np.logical_and(0 <= projected_pts[1, :], projected_pts[1, :] < image.shape[0]))
        mask = np.logical_and(mask, mask_positive)

        masked_pcd = pcd[:, mask]
        image_idx = projected_pts[:, mask]
        label = image[image_idx[1, :], image_idx[0, :]].T  # TODO: figure out the shape of the label

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
        pcd_origin_offset = np.array(
            [[1369.0496826171875], [562.84814453125], [0.0]])  # pcd origin with respect to map origin
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

#
# # The coordinate of the map is defined as map[x, y]
# self.map = None
# self.map_pose = None
# self.save_map_to_file = False
# self.map_boundary = cfg.MAPPING.BOUNDARY
# self.resolution = cfg.MAPPING.RESOLUTION
# self.label_names = cfg.LABELS_NAMES
# self.label_colors = np.array(cfg.LABEL_COLORS)
#
# self.position_rel = np.array([[0, 0, 0]]).T
# self.yaw_rel = 0
#
#
# # This is a testing parameter, when the time stamp reach this number, the entire node will terminate.
# self.test_cut_time = cfg.TEST_END_TIME
#
# if cfg.MAPPING.CONFUSION_MTX.LOAD_PATH != "":
#     confusion_matrix = ConfusionMatrix(load_path=cfg.MAPPING.CONFUSION_MTX.LOAD_PATH)
#     self.confusion_matrix = confusion_matrix.get_submatrix(cfg.LABELS, to_probability=True, use_log=True)
# else:
#     # Use Identity confusion matrix
#     self.confusion_matrix = np.eye(len(self.label_names))
#
# # Print the configuration to user
# self.logger.log("Running with configuration:\n" + str(cfg))
#
# self.ground_truth_dir = cfg.GROUND_TRUTH_DIR
# self.input_list = []
# self.unique_input_dict = {}
# self.input_dir = cfg.MAPPING.INPUT_DIR
