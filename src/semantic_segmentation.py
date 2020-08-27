""" Semantic Segmentation Class

Author: Qinru
Date:February 14, 2020
"""

# module
from __future__ import absolute_import, division, print_function, unicode_literals  # python2 compatibility
# parameters

import os.path as osp

# classes
from src.semantic_segmentation_node import SemanticSegmentation


# functions


# main
def main():
    pass


if __name__ == "__main__":
    main()
    print(osp.dirname(__file__))
    from network.deeplab_v3_plus.config.demo import cfg

    config_file = '../config/avl.yaml'
    cfg.merge_from_file(config_file)

    network = SemanticSegmentation(cfg)

    import PIL.Image as Image
    import matplotlib.pyplot as plt
    import numpy as np

    image = Image.open(
        # "/home/qinru/avl/semantic/deeplab/data/mapillary-vistas-dataset_public_v1.1_processed/validation/images/0B5qssoIEl6LguVQjoRiDQ.jpg"
        "/home/henry/Documents/projects/pylidarmot/src/vision_semantic_segmentation/high_res_images/1.jpg"
    )
    image = np.array(image)

    segmentation = network.segmentation(image)

    plt.imshow(segmentation)
    plt.show()
