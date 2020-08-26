This document will give you the overall idea of the project and the functionalities of each part of it.

we propose to fuse image and pre-built point cloud map information to perform automatic and accurate labeling of static landmarks such as roads, sidewalks, crosswalks, and lanes. The goal is to generate probabilistic semantic maps for road feature extraction and HD mapping applications. Our model consists of three parts: 
1. Semantic Segmentation: predict semantic labels based on 2D images. 
2. Semantic Association: associate point clouds with predicted semantic labels. 
3. Semantic Mapping: use a probabilistic semantic mapping method to capture the latent distribution of each label and utilize LiDAR intensity to augment lane mark prediction.

## Semantic Segmentation
Adapted from deeplab_v3+.

related files:
- src/vision_semantic_segmentation_node.py:
  - ROS communication for image data.
  - Resize the images to speed up processing.
- src/semantic_segmentation.py: handles the network
- src/network/: the deeplab_v3 library

## Semantic Association and Semantic Mapping
The two functionalities are combined in the same file

related files:
- src/mapping.py:
  - ROS communication, subscribe pose, semantic image and local_point_cloud, publish semantic point cloud and semantic map.
  - Project the point cloud into semnatic mask to get semantic labels.
  - Update the semantic map using semantic point cloud with intensity and confusion matrix. 
  - Record hicle file for replay in mapping_replay.py
- src/mapping_replay.py:
  - run the mapping process from deterministic inputs recorded in a hicle file.
  - Functionalities are the same as mapping.py but does not require ROS.
- src/renderer.py: take the probabilistic semantic map and render color images.
