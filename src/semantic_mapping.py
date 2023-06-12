""" Semantic mapping library

Author: Henry Zhang
Date: February 24, 2020

"""
import argparse
import numpy as np
import os.path as osp
import sys

# Add src directory into the path
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "../")))

from src.camera import camera_setup_1, camera_setup_6
from src.data.confusion_matrix import ConfusionMatrix, adjust_for_mapping
from src.homography import generate_homography
from src.renderer import render_bev_map, render_bev_map_with_thresholds, apply_filter
from src.utils.utils import homogenize, dehomogenize, get_rotation_from_angle_2d
from src.utils.utils_ros import get_transformation, get_transform_from_pose, create_point_cloud
from src.utils.logger import MyLogger
from src.utils.file_io import makedirs
from test.test_semantic_mapping import Test

# Dependencies to be removed
from tf.transformations import euler_matrix


class SemanticMapping:
    """
    Create a semantic bird's eye view map from the LiDAR sensor and 2D semantic segmentation image. The BEV map is
    represented by a grid.

    """

    def __init__(self, cfg, logger=None):
        """

        Args:
            cfg: Configuration file
        """
        # Set up the logger
        self.logger = logger

        self.pcd_range_max = cfg.MAPPING.PCD.RANGE_MAX
        self.use_pcd_intensity = cfg.MAPPING.PCD.USE_INTENSITY

        # The coordinate of the map is defined as map[x, y]
        self.map = None
        self.map_boundary = cfg.MAPPING.BOUNDARY
        if cfg.MAPPING.PREV_MAP != "":
            self.load_prev_map(cfg.MAPPING.PREV_MAP)
        
        self.resolution = cfg.MAPPING.RESOLUTION
        self.label_names = cfg.LABELS_NAMES
        self.label_colors = np.array(cfg.LABEL_COLORS)
        self.color_remap_source = np.array(cfg.COLOR_REMAP_SOURCE)
        self.color_remap_dest = np.array(cfg.COLOR_REMAP_DEST)

        self.map_height = int(abs(self.map_boundary[0][1] - self.map_boundary[0][0]) / self.resolution)
        self.map_width = int(abs(self.map_boundary[1][1] - self.map_boundary[1][0]) / self.resolution)
        self.map_depth = len(self.label_names)

        self.position_rel = np.array([[0, 0, 0]]).T
        self.yaw_rel = 0

        self.preprocessing()

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
    
    def load_prev_map(self, prev_map_path):
        print("------------------------------------------------")
        print("Restoring map: " + str(prev_map_path))
        self.map = np.load(prev_map_path)
        print("------------------------------------------------")

    def preprocessing(self):
        """ Setup constant matrices """
        self.T_velodyne_to_basklink = self.set_velodyne_to_baselink()


    def set_velodyne_to_baselink(self):
        print("velodyne to baselink from TF is tunned, current version fits best.")
        T = euler_matrix(0., 0.140, 0.)
        # T = euler_matrix(0., 0.10, 0.)
        t = np.array([[2.64, 0, 1.98]]).T
        T[0:3, -1::] = t
        return T
    

    def mapping(self, 
                pcd, 
                pcd_frame_id, 
                semantic_image, 
                projection_matrix, 
                T_base_to_origin=None, 
                T_local_to_base=None):
        """
        Receives the semantic segmentation image, the pose of the vehicle, and the calibration of the camera,
        we will build a semantic point cloud, then project it into the 2D bird's eye view coordinates.

        Args:
            semantic_image: 2D semantic image
            pose: vehicle pose
            projection_matrix: The 3 by 4 projection matrix of the camera
        """
        # Initialize the map
        if self.map is None:
            self.map = np.zeros((self.map_height, self.map_width, self.map_depth))

        if self.depth_method is 'points_map' or self.depth_method is 'points_raw':
            pcd_in_range, pcd_label = self.project_pcd(pcd, pcd_frame_id, semantic_image,
                                                       projection_matrix, T_base_to_origin)
            pcd_label = self.merge_color(pcd_label)
            # pcd_pub = create_point_cloud(pcd_in_range[0:3].T, pcd_label.T, frame_id=self.pcd_frame_id)
            self.map = self.update_map(self.map, pcd_in_range, pcd_label)
        else:
            self.map = self.update_map_planar(self.map, semantic_image, projection_matrix, T_local_to_base)
        
        return self.map, pcd_in_range, pcd_label
    

    def get_color_map(self, map):
        map_filtered = apply_filter(map)  # smooth the labels to fill black holes
        color_map = render_bev_map(map_filtered, self.label_colors)
        return color_map


    def project_pcd(self, pcd, pcd_frame_id, image, projection_matrix, T_base_to_origin=None):
        """
        Extract labels of each point in the pcd from image
        Args:
            projection_matrix:  the camera projection matrix.

        Returns: Point cloud that are visible in the image, and their associated labels

        """
        if pcd is None:
            return
        
        if pcd_frame_id != "velodyne":
            if T_base_to_origin is None:
                print("Error: T_base_to_origin should not be None. if pcd frame is velodyne")
            T_origin_to_velodyne = np.linalg.inv(np.matmul(T_base_to_origin, self.T_velodyne_to_basklink))

            pcd_velodyne = np.matmul(T_origin_to_velodyne, homogenize(pcd[0:3, :]))
        else:
            pcd_velodyne = homogenize(pcd[0:3, :])

        IXY = dehomogenize(np.matmul(projection_matrix, pcd_velodyne)).astype(np.int32)

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
    

    def merge_color(self, pcd_label):
        # print(pcd_label.shape)
        for color_src, color_dest in zip(self.color_remap_source, self.color_remap_dest):
            pcd_mask = np.all(pcd_label == color_src.reshape(3,1), axis=0)
            pcd_label[:,pcd_mask] = color_dest.reshape(3,1)
        return pcd_label


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
        #pcd_origin_offset = np.array([[1369.0496826171875], [562.84814453125], [0.0]]) # pcd origin with respect to map origin
        pcd_origin_offset = np.array([[abs(self.map_boundary[0][0])], [abs(self.map_boundary[1][0])], [0.0]]) # pcd origin with respect to map origin
        pcd_local = pcd[0:3] + pcd_origin_offset
        pcd_on_map = pcd_local - np.matmul(normal, np.matmul(normal.T, pcd_local))
        # Discretize point cloud into grid, Note that here we are basically doing the nearest neighbor search
        #pcd_pixel = ((pcd_on_map[0:2, :] - np.array([[self.map_boundary[0][0]], [self.map_boundary[1][0]]]))
        #             / self.resolution).astype(np.int32)
        pcd_pixel = (pcd_on_map[0:2, :] / self.resolution).astype(np.int32)
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
                # map[pcd_pixel[0, idx_mask], pcd_pixel[1, idx_mask], i] += 100

                # For the region where there is no intensity by our network detected as lane, we will degrade its
                # threshold
                # non_intensity_mask = np.logical_and(~intensity_mask, idx_mask)
                # map[pcd_pixel[1, non_intensity_mask], pcd_pixel[0, non_intensity_mask], i] -= 0.5

        return map


    def update_map_planar(self, map_local, image, projection_matrix, T_local_to_base):
        """ Project the semantic image onto the map plane and update it """

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
        points_image = dehomogenize(np.matmul(projection_matrix, points_velodyne)).astype(np.int)

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
    pass


if __name__ == "__main__":
    main()
