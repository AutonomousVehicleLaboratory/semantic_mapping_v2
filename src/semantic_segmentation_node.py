#!/usr/bin/env python
""" Semantic Segmentation Ros Wrapper

Author: Hengyuan Zhang
Date:February 14, 2020
"""

from __future__ import absolute_import, division, print_function, unicode_literals  # python2 compatibility

import argparse
import cv2
import numpy as np
import os.path as osp
import rospy
import sys
import torch
import torchvision.transforms as T

# Add src directory into the path
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "../")))

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
from deeplab.data.utils import mapillary_visualization
from deeplab.models.build import build_model
from deeplab.utils.torch_util import set_random_seed

from src.camera import build_camera_model
from src.config.mapping import get_cfg_defaults


class SemanticSegmentation():
    """ Semantic segmentation network """

    def __init__(self, cfg):
        """

        Args:
            cfg: network configuration, please refer to deeplab.config.deeplab_v3_plus
        """
        self.model = build_model(cfg)[0]
        self.model = torch.nn.DataParallel(self.model).cuda()

        # Load weight
        checkpoint = torch.load(cfg.MODEL.WEIGHT, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint.pop('model'))

        # Build transform
        self.transform = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), True),
        ])

    def __call__(self, image_in):
        """
        Generate the semantic segmentation from the input image
        Args:
            image_in: numpy array (h, w, 3) in RGB

        Returns:
            the semantic segmentation mask

        """
        self.model.eval()
        with torch.no_grad():
            image_tensor = self.transform(image_in)
            image_tensor = image_tensor.unsqueeze(dim=0).cuda()
            preds = self.model(image_tensor, upsample_pred=False)
            preds = torch.argmax(preds, dim=1).squeeze().cpu().numpy()
            return preds


class SemanticSegmentationNode:
    """ ROS node for generating semantic segmentation image """

    def __init__(self, cfg):
        if cfg.SEM_SEG.IMAGE_SCALE < 0 or cfg.SEM_SEG.IMAGE_SCALE > 1:
            raise ValueError("image scale should be in the range of [0, 1]")

        self._target_cameras = cfg.SEM_SEG.TARGET_CAMERAS
        queue_size = 1  # If needed, consider put it into config

        # Build ROS subscriber and publisher
        self._img_subscriber = {}
        for cam in self._target_cameras:
            self._img_subscriber[cam] = rospy.Subscriber(
                "/avt_cameras/{}/image_color/compressed".format(cam),
                CompressedImage,
                self._image_callback,
                queue_size=queue_size,
                callback_args="compressed"
            )

        self._img_publisher = {}
        for cam in self._target_cameras:
            self._img_publisher[cam] = rospy.Publisher("/{}/semantic".format(cam), Image, queue_size=queue_size)

        # Collect the camera models
        self.camera_models = {}
        for cam in self._target_cameras:
            self.camera_models[cam] = build_camera_model(cam)

        # Setup the semantic segmentation network
        network_cfg = cfg.SEM_SEG.NETWORK
        self.network = SemanticSegmentation(network_cfg)
        self.semantic_color_fn = mapillary_visualization.apply_color_map
        self.semantic_color_reference = mapillary_visualization.get_labels(network_cfg.DATASET.CONFIG_PATH)
        self.input_image_scale = cfg.SEM_SEG.IMAGE_SCALE

        self.bridge = CvBridge()

    def _image_callback(self, msg, img_type="compressed"):
        camera_name = msg.header.frame_id
        if camera_name not in self._target_cameras:
            rospy.logwarn("Unknown camera %s, no semantic image is generated ", camera_name)
            return

        if img_type == "compressed":
            np_arr = np.fromstring(msg.data, np.uint8)
            # Note that the output of the function below is BGR,
            # reference https://stackoverflow.com/questions/52494592/wrong-colours-with-cv2-imdecode-python-opencv
            image_in = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        elif img_type == "image_color":
            try:
                image_in = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")  # output is BGR
            except CvBridgeError as e:
                print(e)
                return
        else:
            raise NotImplementedError

        image_in = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB)
        image_in = cv2.undistort(image_in, self.camera_models[camera_name].K, self.camera_models[camera_name].dist)
        image_height, image_width = image_in.shape[:2]  # Collect the original size of the image_in

        # Resize the image to reduce the memory overhead
        if self.input_image_scale < 1:
            width = int(image_in.shape[1] * self.input_image_scale)
            height = int(image_in.shape[0] * self.input_image_scale)
            dim = (width, height)
            image_in = cv2.resize(image_in, dim, interpolation=cv2.INTER_AREA)

        # Do semantic segmentation
        image_out = self.network(image_in)
        image_out = image_out.astype(np.uint8)

        # NOTE: we use INTER_NEAREST because values are discrete labels
        if self.input_image_scale:
            image_out = cv2.resize(image_out, (image_width, image_height), interpolation=cv2.INTER_NEAREST)

        # Convert the semantic label to color for visualization
        image_out_colored = self.semantic_color_fn(image_out, self.semantic_color_reference)
        image_out_colored = np.squeeze(image_out_colored)
        image_out_colored = image_out_colored.astype(np.uint8)
        # Convert it to BGR as requested by opencv
        image_out_colored = cv2.cvtColor(image_out_colored, cv2.COLOR_RGB2BGR)

        # Publish the semantic image and its colored version
        try:
            image_for_publish = self.bridge.cv2_to_imgmsg(image_out_colored, encoding="passthrough")
            image_for_publish.header.stamp = msg.header.stamp
            image_for_publish.header.frame_id = msg.header.frame_id
            self._img_publisher[camera_name].publish(image_for_publish)
        except CvBridgeError as e:
            print(e)


def parse_args():
    """ Parse the command line arguments """
    parser = argparse.ArgumentParser(description='Semantic Segmentation ROS Node')
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
    rospy.init_node('semantic_segmentation')

    cfg = get_cfg_defaults()
    args = parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    if cfg.RNG_SEED > -1:
        set_random_seed(cfg.RNG_SEED)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    ros_node = SemanticSegmentationNode(cfg)
    rate = rospy.Rate(30)  # Note that from rosbag, the image comes at 12Hz

    while not rospy.is_shutdown():
        rate.sleep()


if __name__ == '__main__':
    main()
