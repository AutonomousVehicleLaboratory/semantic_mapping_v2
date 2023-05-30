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
from src.node_config.base_cfg import get_cfg_defaults
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
        self.input_dir = cfg.MAPPING.INPUT_DIR
        self.round_close = cfg.MAPPING.ROUND_CLOSE

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
        T = euler_matrix(0., 0.140, 0.)
        t = np.array([[2.64, 0, 1.98]]).T
        T[0:3, -1::] = t
        return T

    def mapping_replay_dir(self):
        """ Replay all input hkl files in the input_dir directory 
            Please specify the input_dir in the config file.
        """
        if os.path.exists(self.input_dir):
            for file_name in os.listdir(self.input_dir):
                if file_name.endswith('.hkl'):
                    hkl_file =  os.path.join(self.input_dir, file_name)
                    print("Loading input file " + hkl_file)
                    with open(hkl_file, 'rb') as hkl_file_pointer:
                        input_list = hickle.load(hkl_file_pointer)
                        print("Hkl file loaded!")
                        self.mapping_replay(input_list, file_name[0:-4])
                        hkl_file_pointer.close()
        
        if self.ground_truth_dir != "":
            test = Test(ground_truth_dir=self.ground_truth_dir, logger=self.logger)
            test.full_test(dir_path=self.output_dir, latex_mode=True)

    def mapping_replay_file(self):
        """ Replay a single hkl file in the input_dir directory
            Please specify the input_dir in the config file.
        """
        # load files
        file_name = "input_list_0.hkl"
        hkl_file = os.path.join(self.input_dir, file_name)
        print("Loading input file " + hkl_file )
        with open(hkl_file, 'rb') as hkl_file_pointer:
            input_list = hickle.load(hkl_file_pointer)
            print("Hkl file loaded!")
            self.mapping_replay(input_list, file_name[0:-4])


    def mapping_replay(self, input_list, file_name):
        """ Map the given input to a semantic global map
            input_list: a list of data frames, each frame is a dictionary stores the data
            file_name: name of the input file, append to global map to distinguish them.
        """
        # Initialize the map
        self.map = np.zeros((self.map_height, self.map_width, self.map_depth))
        camera_calibration = self.cam1

        for frame_input_dict in input_list:
            pcd = frame_input_dict["pcd"]
            pcd_frame_id = frame_input_dict["pcd_frame_id"]
            semantic_image = frame_input_dict["semantic_image"]
            pose = frame_input_dict["pose"]
            pcd_in_range, pcd_label = self.project_pcd(pcd, pcd_frame_id, semantic_image, pose,
                                                       camera_calibration)

            self.map = self.update_map(self.map, pcd_in_range, pcd_label)

        output_dir = self.output_dir
        makedirs(output_dir, exist_ok=True)
        # np.save(osp.join(output_dir, "map.npy"), self.map)

        self.map = apply_filter(self.map)  # smooth the labels to fill black holes

        color_map = render_bev_map(self.map, self.label_colors)
        # color_map = render_bev_map_with_thresholds(self.map, self.label_colors, priority=[3, 4, 0, 2, 1],
        #                                            thresholds=[0.1, 0.1, 0.5, 0.20, 0.05])

        output_file = osp.join(output_dir, "global_map_" + file_name + ".png")
        print("Saving image to", output_file)
        cv2.imwrite(output_file, color_map)

        # evaluate
        if self.ground_truth_dir != "":
            test = Test(ground_truth_dir=self.ground_truth_dir, logger=self.logger)
            test.test_single_map(color_map)


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

        if self.round_close == True:
            IXY = np.around( dehomogenize(np.matmul(camera_calibration.P, pcd_velodyne))).astype(np.int32)
        else:
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
        if self.round_close == True:
            pcd_pixel = np.around((pcd_on_map[0:2, :] - np.array([[self.map_boundary[0][0]], [self.map_boundary[1][0]]]))
                         / self.resolution).astype(np.int32)
        else:
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
    args = parser.parse_args(sys.argv[1::])
    return args


def main():
    cfg = get_cfg_defaults()
    args = parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    
    sm = SemanticMapping(cfg)
    # sm.mapping_replay_file()
    sm.mapping_replay_dir()
    

if __name__ == "__main__":
    main()
