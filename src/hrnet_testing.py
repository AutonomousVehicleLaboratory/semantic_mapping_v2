from src.hrnet.hrnet_semantic_segmentation import HRNetSemanticSegmentation, get_custom_hrnet_args

if __name__ == "__main__":
    seg = HRNetSemanticSegmentation(get_custom_hrnet_args())

    mask_out = self.seg.segmentation(input_img)
    mask_out = mask_out.astype(np.uint8).squeeze()

    colored_ouptput = self.seg_color_fn(mask_out)
