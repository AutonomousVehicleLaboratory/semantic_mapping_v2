"""
python hrnet_semantic_segmentation.py --num_workers 1 --dataset=mapillary --cv=0 --bs_val=1 --eval=folder --eval_folder='./imgs/test_imgs' --dump_assets --dump_all_images --n_scales="0.5,1.0,2.0" --snapshot="ASSETS_PATH/seg_weights/mapillary_ocrnet.HRNet_Mscale_fast-rattlesnake.pth" --arch=ocrnet.HRNet_Mscale --result_dir=LOGDIR

Copyright 2020 Nvidia Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import absolute_import
from __future__ import division

import argparse
import os
import sys
import time
import torch

from runx.logx import logx
from hrnet.config import assert_and_infer_cfg, update_epoch, cfg
from hrnet.utils.misc import AverageMeter, prep_experiment, eval_metrics
from hrnet.utils.misc import ImageDumper
from hrnet.utils.trnval_utils import eval_minibatch, validate_topn
from hrnet.loss.utils import get_loss
from hrnet.loss.optimizer import get_optimizer, restore_opt, restore_net
from PIL import Image
from hrnet.default_parser import args as def_args
import numpy as np
import torchvision.transforms as standard_transforms

import hrnet.datasets as datasets
import hrnet.network as network

def get_custom_hrnet_args():
    def_args.num_workers=1
    def_args.dataset='mapillary'
    def_args.cv=0
    def_args.bs_val=1
    # def_args.eval='folder'
    # def_args.eval_folder='./imgs/test_imgs'
    def_args.dump_assets=True
    def_args.dump_all_images=True
    # def_args.n_scales="0.5,1.0,2.0"
    def_args.n_scales="0.25,0.5,1.0"
    # def_args.n_scales="1.0"
    def_args.snapshot="ASSETS_PATH/seg_weights/mapillary_ocrnet.HRNet_Mscale_fast-rattlesnake.pth"
    def_args.arch='ocrnet.HRNet_Mscale'
    def_args.result_dir='LOGDIR'

    return def_args

def main():
    # Example usage
    HRNet = HRNetSemanticSegmentation(get_custom_hrnet_args())
    # img = Image.open("./imgs/test_imgs/scholars_far_resize.jpg").convert('RGB')
    img = Image.open("/home/hzhang/data/hrnet/test.png").convert('RGB')
    img.load()
    img = np.asarray(img, dtype='uint8')
    print(HRNet.segmentation(img))

class HRNetSemanticSegmentation():
    def __init__(self, args):
        """

        Args:
            args: network configuration
        """
        # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
        assert_and_infer_cfg(args)

        # train_loader, val_loader, train_obj = \
        #     datasets.setup_loaders(args)
        criterion, criterion_val = get_loss(args)

        if args.snapshot:
            if 'ASSETS_PATH' in args.snapshot:
                args.snapshot = args.snapshot.replace('ASSETS_PATH', cfg.ASSETS_PATH)
            checkpoint = torch.load(args.snapshot,
                                    map_location=torch.device('cpu'))
            args.restore_net = True
            msg = "Loading weights from: checkpoint={}".format(args.snapshot)
            print(msg)

        self.net = network.get_net(args, criterion)
        optim, scheduler = get_optimizer(args, self.net)

        self.net = network.wrap_network_in_dataparallel(self.net, args.apex)

        if args.summary:
            print(str(self.net))
            from pytorchOpCounter.thop import profile
            img = torch.randn(1, 3, 1024, 2048).cuda()
            mask = torch.randn(1, 1, 1024, 2048).cuda()
            macs, params = profile(self.net, inputs={'images': img, 'gts': mask})
            print(f'macs {macs} params {params}')
            sys.exit()

        if args.restore_optimizer:
            restore_opt(optim, checkpoint)
        if args.restore_net:
            restore_net(self.net, checkpoint)

        if args.init_decoder:
            self.net.module.init_mods()

        torch.cuda.empty_cache()

        if args.start_epoch != 0:
            scheduler.step(args.start_epoch)
        
        self.args = args
                    
    def inference(self, image, criterion, optim, epoch,
                calc_metrics=True,
                dump_all_images=False):
        """
        Run inference on a single image

        :image: (1,3,h,w) pytorch tensor of image
        :net: the network
        :criterion: loss fn
        :optimizer: optimizer
        :epoch: current epoch
        :calc_metrics: calculate validation score
        :dump_all_images: dump all images, not just N
        """

        dump_assets = self.args.dump_assets

        self.net.eval()
        val_loss = AverageMeter()

        val_input_transform = standard_transforms.Compose([
            standard_transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD)
        ])

        image_normalized = val_input_transform(image)

        labels_zero = torch.zeros(1, image_normalized.shape[1], image_normalized.shape[2])
        filename = 'Random_Test'
        single_value = torch.ones(1)

        #Convert image to batched
        image_normalized = image_normalized.unsqueeze(0)

        data = (
            image_normalized,
            labels_zero,
            filename,
            single_value
        )

        # Run network
        assets, _ = \
            eval_minibatch(data, self.net, criterion, val_loss, calc_metrics,
                            self.args, 0)

        input_images, labels, img_names, _ = data
        
        return assets['predictions']
    
    def segmentation(self, img):
        """
        Run semantic segmentation on a single numpy image in form
        (h, w, 3) where it is uint8 0-255
        """

        tensor_transform = standard_transforms.ToTensor()
        image_data = tensor_transform(img)

        return self.inference(image_data, criterion=None, optim=None, epoch=0,
                    calc_metrics=False, dump_all_images=True)
    


if __name__ == "__main__":
    main()