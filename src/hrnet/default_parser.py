import argparse

args = argparse.Namespace()

args.lr=0.002
args.arch='deepv3.DeepWV3Plus'
"""Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
and deepWV3Plus (backbone: WideResNet38)."""
args.dataset='cityscapes'
"""cityscapes, mapillary, camvid, kitti"""
args.dataset_inst=None
"""placeholder for dataset instance"""
args.num_workers=4
"""cpu worker threads per dataloader instance"""

args.cv=0
"""Cross-validation split id to use. Default # of splits set'
' to 3 in config"""

args.class_uniform_pct=0.5
"""What fraction of images is uniformly sampled"""
args.class_uniform_tile=1024
"""tile size for class uniform sampling"""
args.coarse_boost_classes=None
"""Use coarse annotations for specific classes"""

args.custom_coarse_dropout_classes=None
"""Drop some classes from auto-labelling"""

args.img_wt_loss=False
"""per-image class-weighted loss"""
args.rmi_loss=False
"""use RMI loss"""
args.batch_weighting=False
"""Batch weighting for class (use nll class weighting using '
'batch stats"""

args.jointwtborder=False
"""Enable boundary label relaxation"""
args.strict_bdr_cls=''
"""Enable boundary label relaxation for specific classes"""
args.rlx_off_epoch=-1
"""Turn off border relaxation after specific epoch count"""
args.rescale=1.0
"""Warm Restarts new lr ratio compared to original lr"""
args.repoly=1.5
"""Warm Restart new poly exp"""

args.apex=False
"""Use Nvidia Apex Distributed Data Parallel"""
args.fp16=False
"""Use Nvidia Apex AMP"""

args.local_rank=0
"""parameter used by apex library"""
args.global_rank=0
"""parameter used by apex library"""

args.optimizer='sgd'
"""optimizer"""
args.amsgrad=False
"""amsgrad for adam"""

args.freeze_trunk=False
args.hardnm=0
"""0 means no aug, 1 means hard negative mining '
'iter 1, 2 means hard negative mining iter 2"""

args.trunk='resnet101'
"""trunk model, can be: resnet101 (default), resnet50"""
args.max_epoch=180
args.max_cu_epoch=150
"""Class Uniform Max Epochs"""
args.start_epoch=0
args.color_aug=0.25
"""level of color augmentation"""
args.gblur=False
"""Use Guassian Blur Augmentation"""
args.bblur=False
"""Use Bilateral Blur Augmentation"""
args.brt_aug=False
"""Use brightness augmentation"""
args.lr_schedule='poly'
"""name of lr schedule: poly"""
args.poly_exp=1.0
"""polynomial LR exponent"""
args.poly_step=110
"""polynomial epoch step"""
args.bs_trn=2
"""Batch size for training per gpu"""
args.bs_val=1
"""Batch size for Validation per gpu"""
args.crop_size='896'
"""training crop size: either scalar or h,w"""
args.scale_min=0.5
"""dynamically scale training images down to this size"""
args.scale_max=2.0
"""dynamically scale training images up to this size"""
args.weight_decay=1e-4
args.momentum=0.9
args.snapshot=None
args.resume=None
"""continue training from a checkpoint. weights, '
'optimizer, schedule are restored"""
args.restore_optimizer=False
args.restore_net=False
args.exp='default'
"""experiment directory name"""
args.result_dir='./logs'
"""where to write log output"""
args.syncbn=False
"""Use Synchronized BN"""
args.dump_augmentation_images=False
"""Dump Augmentated Images for sanity check"""
args.test_mode=False
"""Minimum testing to verify nothing failed, '
'Runs code for 1 epoch of train and val"""
args.wt_bound=1.0
"""Weight Scaling for the losses"""
args.maxSkip=0
"""Skip x number of  frames of video augmented dataset"""
args.scf=False
"""scale correction factor"""
# Full Crop Training
args.full_crop_training=False
"""Full Crop Training"""

# Multi Scale Inference
args.multi_scale_inference=False
"""Run multi scale inference"""

args.default_scale=1.0
"""default scale to run validation"""

args.log_msinf_to_tb=False
"""Log multi-scale Inference to Tensorboard"""

args.eval=None
"""just run evaluation, can be set to val or trn or '
'folder"""
args.eval_folder=None
"""path to frames to evaluate"""
args.three_scale=False
args.alt_two_scale=False
args.do_flip=False
args.extra_scales='0.5,2.0'
args.n_scales=None
args.align_corners=False
args.translate_aug_fix=False
args.mscale_lo_scale=0.5
"""low resolution training scale"""
args.pre_size=None
"""resize long edge of images to this before'
' augmentation"""
args.amp_opt_level='O1'
"""amp optimization level"""
args.rand_augment=None
"""RandAugment setting: set to \'N,M\'"""
args.init_decoder=False
"""initialize decoder with kaiming normal"""
args.dump_topn=0
"""Dump worst val images"""
args.dump_assets=False
"""Dump interesting assets"""
args.dump_all_images=False
"""Dump all images, not just a subset"""
args.dump_for_submission=False
"""Dump assets for submission"""
args.dump_for_auto_labelling=False
"""Dump assets for autolabelling"""
args.dump_topn_all=False
"""dump topN worst failures"""
args.custom_coarse_prob=None
"""Custom Coarse Prob"""
args.only_coarse=False
args.mask_out_cityscapes=False
args.ocr_aspp=False
args.map_crop_val=False
args.aspp_bot_ch=None
args.trial=None
args.mscale_cat_scale_flt=False
args.mscale_dropout=False
args.mscale_no3x3=False 
"""no inner 3x3"""
args.mscale_old_arch=False 
"""use old attention head"""
args.mscale_init=None
"""default attention initialization"""
args.attnscale_bn_head=False
args.set_cityscapes_root=None
"""override cityscapes default root dir"""
args.ocr_alpha=None
"""set HRNet OCR auxiliary loss weight"""
args.val_freq=1
"""how often (in epochs) to run validation"""
args.deterministic=False
args.summary=False
args.segattn_bot_ch=None
"""bottleneck channels for seg and attn heads"""
args.grad_ckpt=False
args.no_metrics=False
"""prevent calculation of metrics"""
args.supervised_mscale_loss_wt=None
"""weighting for the supervised loss"""

args.ocr_aux_loss_rmi=False
"""allow rmi for aux loss"""