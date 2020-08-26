# This file will study the label of the semantic segmentation network
import json


def read_json_file(filename: str):
    with open(filename) as json_file:
        data = json.load(json_file)
    return data


if __name__ == '__main__':
    data = read_json_file("/home/qinru/avl/playground/vision_semantic_segmentation/external_data/config.json")
    labels = data["labels"]
    for idx, label in enumerate(labels):
        print("idx {:<3} label {:>30} color {:<15}".format(idx, label['readable'], str(label['color'])))
