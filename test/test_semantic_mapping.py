import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def convert_labels(gmap, mask=None):
    """ covert colors to labels """
    if mask is None:
        mask = np.ones((gmap.shape[0], gmap.shape[1]))
    else:
        mask = mask[:gmap.shape[0],:gmap.shape[1]]
    global_map = np.zeros((gmap.shape[0], gmap.shape[1]))
    global_map[np.logical_and(np.all(gmap == np.array([128, 64, 128]), axis=-1), mask)] = 1  # road
    global_map[np.logical_and(np.all(gmap == np.array([140, 140, 200]), axis=-1), mask)] = 2  # crosswalk
    global_map[np.logical_and(np.all(gmap == np.array([255, 255, 255]), axis=-1), mask)] = 3  # lane
    global_map[np.logical_and(np.all(gmap == np.array([244, 35, 232]), axis=-1), mask)] = 4  # sidewalk
    global_map[np.logical_and(np.all(gmap == np.array([107, 142, 35]), axis=-1), mask)] = 5  # vegetation
    return global_map

label_color_dict = {
    1:np.array([128, 64, 128]),
    2:np.array([140, 140, 200]),
    3:np.array([255, 255, 255]),
    4:np.array([244, 35, 232]),
    5:np.array([107, 142, 35]),
}


def color_labels(gmap, mask=None):
    """ covert labels to colors """
    global_map = np.zeros((gmap.shape[0], gmap.shape[1], 3))
    for label in label_color_dict:
        global_map[gmap == label] = label_color_dict[label]
    return global_map

def read_img(global_map_path, mask=None):
    """ read the global map file and covert colors to labels """
    gmap = cv2.imread(global_map_path)
    # gmap = np.rot90(gmap, k=1, axes=(0, 1))
    global_map = convert_labels(gmap, mask)
    return gmap, global_map


class Test:
    def __init__(self, ground_truth_dir="./", shift_h=0, shift_w=0, logger=None):
        """
            Load the ground truth map and do transformations for it. Preprocess and store it for faster testing.
            ground_truth_dir: dir path to ground truth map
            preprocess: reprocess the rgb ground truth map to interger label if true.
        """
        truth_file_path = os.path.join(ground_truth_dir, "truth.npy")
        mask_file_path = os.path.join(ground_truth_dir, "mask.npy")

        if os.path.exists(truth_file_path) and os.path.exists(mask_file_path):
            print(truth_file_path, "and", mask_file_path, "exists, openning it.")
            with open(truth_file_path, 'rb') as f:
                self.ground_truth_mask = np.load(f)
            with open(mask_file_path, 'rb') as f:
                self.mask = np.load(f)
        else:
            # preprocess to generate the file
            print(truth_file_path, "does not exist, preprocess the ground truth to generate it.")
            crosswalks = cv2.imread(os.path.join(ground_truth_dir, "bev-5cm-crosswalk.jpg"))
            road = cv2.imread(os.path.join(ground_truth_dir, "bev-5cm-road.jpg"))
            lane = cv2.imread(os.path.join(ground_truth_dir, "bev-5cm-lanes.jpg"))
            mask = cv2.imread(os.path.join(ground_truth_dir, "bev-5cm-mask.jpg"))
            # search if the lane marks have ground truth
            w, h = road.shape[:2]
            # white color in mask corresponds to valid region
            mask = cv2.resize(mask, (int(h / 4), int(w / 4)))
            mask2 = np.zeros((int(w / 4), int(h / 4)))
            # mask2[np.all(mask == np.array([255, 255, 255]), axis=-1)] = 1
            mask2[np.all(mask>255/2, axis=-1)] = 1 
            mask = mask2
            self.mask = mask
            # downsample the image
            crosswalks = cv2.resize(crosswalks, (int(h / 4), int(w / 4)))
            road = cv2.resize(road, (int(h / 4), int(w / 4)))
            lane = cv2.resize(lane, (int(h / 4), int(w / 4)))
            # only use the region within the mask
            self.ground_truth_mask = np.zeros((road.shape[0], road.shape[1]))
            self.ground_truth_mask[np.logical_and(np.any(road > 0, axis=-1), mask)] = 1  # road
            self.ground_truth_mask[np.logical_and(np.any(lane > 0, axis=-1), mask)] = 3  # lanes
            self.ground_truth_mask[np.logical_and(np.any(crosswalks > 0, axis=-1), mask)] = 2  # crosswalk
            with open(truth_file_path, 'wb') as f:
                np.save(f, self.ground_truth_mask)
            with open(mask_file_path, 'wb') as f:
                np.save(f, mask)
        
        self.d = {0: "road", 1: "crosswalk", 2: "lane"}
        self.class_lists = [1, 2, 3]
        self.shift_w = shift_w
        self.shift_h = shift_h
        self.logger = logger
    
    def log(self, content):
        if self.logger is None:
            print(content)
        else:
            self.logger.log(content)

    def full_test(self, dir_path="./global_maps", visualize=False, latex_mode=False, verbose=False):
        """
            test all the generated maps in dir_path folders
            dir_path: dir path to generated maps
        """
        file_lists = os.listdir(dir_path)
        file_lists = sorted([x for x in file_lists if ".png" in x])
        path_lists = [os.path.join(dir_path, x) for x in file_lists]
        iou_array = []
        acc_array = []
        miss_array = []
        for path in path_lists:
            print("You are testing\t" + path)
            _, generate_map = read_img(path, self.mask)
            gmap = self.ground_truth_mask[self.shift_w:generate_map.shape[0] + self.shift_w,
                   self.shift_h:generate_map.shape[1] + self.shift_h]
            iou_lists, acc_lists, miss = self.iou_expand(gmap, generate_map, latex_mode=latex_mode, verbose=verbose)
            iou_array.append(np.array(iou_lists).reshape(1, -1))
            acc_array.append(np.array(acc_lists).reshape(1, -1))
            miss_array.append(miss)
            if visualize:
                mask = np.zeros(generate_map.shape)
                for cls in self.class_lists:
                    mask = np.logical_or(mask, generate_map == cls)
                generate_map[np.logical_not(mask)] = 0
                self.imshow(gmap, generate_map)
                self.disparity(gmap, generate_map)
        miss = np.mean(miss_array)
        miss_percent = miss * 100
        iou_array = np.concatenate(iou_array, axis=0)
        iou_lists = np.mean(iou_array, axis=0)
        acc_array = np.concatenate(acc_array, axis=0)
        acc_lists = np.mean(acc_array, axis=0)
        # self.log("Final Batch evaluation")
        # self.log("IOU for {}: {}\t{}: {}\t{}:{}\tmIOU: {}".format(self.d[0], iou_lists[0], self.d[1], iou_lists[1],
        #                                                        self.d[2], iou_lists[2],
        #                                                        np.mean(iou_lists)))
        # self.log("ACC for {}: {}\t{}: {}\t{}:{}\tmIOU: {}".format(self.d[0], acc_lists[0], self.d[1], acc_lists[1],
        #                                                        self.d[2], acc_lists[2],
        #                                                        np.mean(acc_lists)))
        # self.log("Overall Missing rate: {}".format(miss))
        # if latex_mode:
        #     self.log("&{:.3f}&{:.3f}&{:.3f}&{:.3f}&{:.3f}&{:.3f}&{:.3f}&{:.3f}&{:.3g}\\\\ \\hline".format(iou_lists[0], iou_lists[1], iou_lists[2], np.mean(iou_lists),
        #     acc_lists[0], acc_lists[1], acc_lists[2], np.mean(acc_lists), miss_percent))

    def test_single_map(self, global_map):
        """ Calculate and print the IoU, accuracy and missing rate
            of the global_map and ground truth. 
            global_map: the semantic global map
        """
        generate_map = convert_labels(global_map)
        gmap = self.ground_truth_mask[self.shift_w:generate_map.shape[0] + self.shift_w,
               self.shift_h:generate_map.shape[1] + self.shift_h]
        self.iou(gmap, generate_map, verbose=True)

    def iou(self, gmap, generate_map, latex_mode=False, verbose=False):
        """
            Calculate and print the IoU, accuracy, missing rate
            gmap: ground truth map with interger labels
            generate_map: generated map with interger labels
        """
        iou_lists = []
        acc_lists = []
        for cls in self.class_lists:
            gmap_layer = gmap == cls
            map_layer = generate_map == cls
            intersection = float(np.sum(gmap_layer * map_layer))
            union = float(np.sum(gmap_layer) + np.sum(map_layer) - intersection)
            iou = intersection / union
            iou_lists.append(iou)
            acc = intersection / np.sum(gmap_layer)
            acc_lists.append(acc)
        miss = 1 - np.sum(np.logical_and((gmap > 0), (generate_map > 0))) / float(np.sum(gmap > 0))
        accuracy = np.sum((gmap == generate_map)[gmap > 0]) / float(np.sum(gmap > 0))
        if verbose:
            if not latex_mode:
                self.log("IOU for {}: {}\t{}: {}\t{}:{}\tmIOU: {}".format(self.d[0], iou_lists[0], self.d[1],
                                                                       iou_lists[1],
                                                                       self.d[2], iou_lists[2],
                                                                       np.mean(iou_lists)))
                self.log("Accuracy for {}: {}\t{}: {}\t{}:{}\tmean Accuracy: {}".format(self.d[0], acc_lists[0],
                                                                                     self.d[1], acc_lists[1],
                                                                                     self.d[2],
                                                                                     acc_lists[2],
                                                                                     accuracy))
                self.log("Overall Missing rate: {}".format(miss))
            else:
                miss_percent = miss * 100
                # print(f"&{iou_lists[0]:.3f}&{iou_lists[1]:.3f}&{iou_lists[2]:.3f}&{np.mean(iou_lists):.3f}&{miss_percent:.3g}\\\\ \\hline")
        return iou_lists, acc_lists, miss
    
    def compute_iou(self, gmap_layer, map_layer):
        intersection = float(np.sum(gmap_layer * map_layer))
        tp = intersection
        fp = float(np.sum(np.logical_and(map_layer, np.logical_not(gmap_layer))))
        fn = float(np.sum(np.logical_and(np.logical_not(map_layer), gmap_layer)))
        union = float(np.sum(gmap_layer) + np.sum(map_layer) - intersection)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        iou = intersection / union
        return tp, fp, fn, precision, recall, intersection, union, iou
    
    def dialate_mask(self, mask, kernel_size=3, iteration=1):
        # Taking a matrix of size 3 as the kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_dialated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=iteration) == 1
        return mask_dialated

    def iou_expand(self, gmap, generate_map, latex_mode=False, verbose=False):
        """
            Calculate and print the IoU, accuracy, missing rate
            gmap: ground truth map with interger labels
            generate_map: generated map with interger labels
        """
        iou_lists = []
        acc_lists = []
        precision_lists = []
        recall_lists = []
        
        
        for idx, cls in enumerate(self.class_lists):
            gmap_layer = gmap == cls
            map_layer = generate_map == cls
            
            tp, fp, fn, precision, recall, intersection, union, iou = self.compute_iou(gmap_layer, map_layer)
            # print('TP:', tp, 'FP:', fp, 'FN:', fn, 'Precision:', precision, 'Recall:', recall, 'iou:', iou)
            original_iou = iou
            
            gmap_layer_dialated = self.dialate_mask(gmap_layer, kernel_size=3, iteration=1)
            tp, fp, fn, precision, recall, intersection, union, iou = self.compute_iou(gmap_layer_dialated, map_layer)
            precision_dialated = precision
            # print('groundtruth dialated TP:', tp, 'FP:', fp, 'FN:', fn, 'Precision:', precision, 'Recall:', recall, 'iou:', iou)
            
            map_layer_dialated = self.dialate_mask(map_layer, kernel_size=3, iteration=1)
            tp, fp, fn, precision, recall, intersection, union, iou = self.compute_iou(gmap_layer, map_layer_dialated)
            # print('generated map dialated TP:', tp, 'FP:', fp, 'FN:', fn, 'Precision:', precision, 'Recall:', recall, 'iou:', iou)
            recall_dialated = recall
            # print(self.d[idx], 'dialated Precision:', precision_dialated, 'recall:', recall_dialated, 'iou:', original_iou)
            
            iou_lists.append(round(original_iou,3))
            acc_lists.append(round(recall,3))
            precision_lists.append(round(precision_dialated,3))
            recall_lists.append(round(recall_dialated,3))
            
        miss = 1 - np.sum(np.logical_and((gmap > 0), (generate_map > 0))) / float(np.sum(gmap > 0))
        accuracy = np.sum((gmap == generate_map)[gmap > 0]) / float(np.sum(gmap > 0))
        if verbose:
            if not latex_mode:
                self.log("IOU for {}: {}\t{}: {}\t{}:{}\tmIOU: {}".format(self.d[0], iou_lists[0], self.d[1],
                                                                       iou_lists[1],
                                                                       self.d[2], iou_lists[2],
                                                                       np.mean(iou_lists)))
                self.log("Accuracy for {}: {}\t{}: {}\t{}:{}\tmean Accuracy: {}".format(self.d[0], acc_lists[0],
                                                                                     self.d[1], acc_lists[1],
                                                                                     self.d[2],
                                                                                     acc_lists[2],
                                                                                     accuracy))
                self.log("Overall Missing rate: {}".format(miss))
            else:
                miss_percent = miss * 100
                # print(f"&{iou_lists[0]:.3f}&{iou_lists[1]:.3f}&{iou_lists[2]:.3f}&{np.mean(iou_lists):.3f}&{miss_percent:.3g}\\\\ \\hline")
                # print("&{:.3f}&{:.3f}&{:.3f}&{:.3f}&{:.3f}&{:.3f}&{:.3f}&{:.3f}&{:.3g}\\\\ \\hline".format(iou_lists[0], iou_lists[1], iou_lists[2], np.mean(iou_lists),
            # acc_lists[0], acc_lists[1], acc_lists[2], np.mean(acc_lists), miss_percent))
                print("&{:.3f}&{:.3f}&{:.3f}&{:.3f}&{:.3f}&{:.3f}&{:.3f}&{:.3f}&{:.3f}\\\\".format(
                    precision_lists[0], recall_lists[0], iou_lists[0],
                    precision_lists[1], recall_lists[1], iou_lists[1],
                    precision_lists[2], recall_lists[2], iou_lists[2]
                ))
        return iou_lists, acc_lists, miss

    def disparity(self, gmap, generate_map):
        """ generate disparity map for the specified channels """
        for i in [1,2,3]:
            bg = np.zeros((gmap.shape[0], gmap.shape[1], 3))
            
            gmap_masked = gmap == i
            generate_map_masked = generate_map == i
            tt = np.logical_and(gmap_masked, generate_map_masked)
            fp = np.logical_and(np.logical_not(gmap_masked), generate_map_masked)
            fn = np.logical_and(gmap_masked, np.logical_not(generate_map_masked))

            bg[tt] = np.array([0, 255, 0])
            bg[fp] = np.array([255, 0, 0])
            bg[fn] = np.array([0, 0, 255])

            plt.figure()
            # plt.imshow(bg[500::, 4000::])# [805:885, 5350:5700])
            # plt.imshow(bg[805:885, 5350:5700])
            plt.imshow(bg[805:870, 4850:5200])
            plt.show()

            img_name = "/home/hzhang/Pictures/sensors/disparity_map_vanilla_i_label_{}.png".format(i)
            img_name = "/home/hzhang/Pictures/sensors/disparity_{}.png".format(i)
            cv2.imwrite(img_name, cv2.cvtColor(bg[805:870, 4850:5200].astype(np.uint8), cv2.COLOR_RGB2BGR))
            img_name = "/home/hzhang/Pictures/sensors/prediction_{}.png".format(i)
            cv2.imwrite(img_name, color_labels(generate_map[805:870, 4850:5200]).astype(np.uint8))
            img_name = "/home/hzhang/Pictures/sensors/groundtruth_{}.png".format(i)
            cv2.imwrite(img_name, color_labels(gmap[805:870, 4850:5200]).astype(np.uint8))
        
        # exit(0)
    
    def imshow(self, img1, img2):
        fig, axes = plt.subplots(1, 2)
        axes[0].matshow(img1)
        axes[1].matshow(img2)
        plt.show()


def main():
    visualize = False # True if visualizing global maps and ground truth, default to no visualization
    latex_mode = True # True if generate latex code of tabels
    verbose = True # True if print evaluation results for every image False if print final average result
    import sys

    # add arguement -v for visualization
    if len(sys.argv) > 1:
        if sys.argv[1] == '-v':
            visualize = True

    # dir_path = "/home/henry/Documents/projects/pylidarmot/src/vision_semantic_segmentation/outputs/distance_new/version_3/"
    # dir_path = "/home/henry/Documents/projects/pylidarmot/src/vision_semantic_segmentation/outputs/without_filter/version_2/"
    # dir_path = "/home/henry/Documents/projects/pylidarmot/src/vision_semantic_segmentation/outputs/points_raw/version_1/"
    # dir_path = "/home/henry/Documents/projects/pylidarmot/src/vision_semantic_segmentation/outputs/alignment/version_10/"
    # dir_path = "/home/hzhang/Documents/projects/noeticws/src/vision_semantic_segmentation/outputs/cfn_mtx_with_intensity/version_75"
    # dir_path = "/home/hzhang/Documents/projects/noeticws/src/vision_semantic_segmentation/outputs/hrnet_label_mapping/version_11"
    # dir_path = "./global_maps"
    # dir_path = "/home/hzhang/Documents/projects/noeticws/src/vision_semantic_segmentation/outputs/cfn_mtx_with_intensity/version_93"
    # dir_path = "/home/hzhang/Documents/projects/noeticws/src/vision_semantic_segmentation/outputs/hrnet_label_mapping/version_10"
    # dir_path = "/home/hzhang/Documents/projects/noeticws/src/vision_semantic_segmentation/outputs/deeplabv3plus_results/version_2"
    dir_path = "/home/hzhang/Documents/projects/noeticws/src/vision_semantic_segmentation/outputs/results"

    ground_truth_dir = "/home/hzhang/data/semantic_mapping/groundtruth"
    
    test = Test(ground_truth_dir=ground_truth_dir)
    test.full_test(dir_path=dir_path, visualize=visualize, latex_mode=latex_mode, verbose=verbose)


def disparity_of_disparity():
    for i in [1,2,3]:
        vanilla_name = "/home/henry/Pictures/IROS/disparity_map_vanilla_i_label_{}.png".format(i)
        cfn_name = "/home/henry/Pictures/IROS/disparity_map_cfn_i_label_{}.png".format(i)

        vanilla_img = cv2.cvtColor( cv2.imread(vanilla_name), cv2.COLOR_BGR2RGB)
        cfn_img = cv2.cvtColor( cv2.imread(cfn_name), cv2.COLOR_BGR2RGB)

        for j in [0, 1, 2]:
            bg = np.zeros((vanilla_img.shape[0], vanilla_img.shape[1], 3))
            bg_layer = bg[:,:,j]
            mask_j = vanilla_img[:,:,j] != cfn_img[:,:,j]
            bg_layer[mask_j] = 255
            
            img_name = "/home/henry/Pictures/IROS/disparity_of_disparity_map_{}_{}.png".format(i, j)
            cv2.imwrite(img_name , cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_RGB2BGR))

        # plt.figure()
        # plt.imshow(bg)# [805:885, 5350:5700])
        # plt.show()

        
        

if __name__ == "__main__":
    main()
    # disparity_of_disparity()
    
