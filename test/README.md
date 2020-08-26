# Test Module
* IOU
* Accuracy
* Missing Rate

## Setup


Please put the generated maps that you want to test in the `./global_maps` directory. 
Please put ground truth maps  (`bev-5cm-crosswalks.jpg`, `bev-5cm-road.jpg`. etc.) in `./ground_truth` directory. 

The file tree should look like this:

```bash
.
├── README.md
├── global_maps
│   ├── bev-5cm-crosswalk.jpg
│   ├── bev-5cm-lanes.jpg
│   ├── bev-5cm-mask.jpg
│   └── bev-5cm-road.jpg
├── ground_truth
│   └── generated_map1.png
│   └── generated_map2.png
├── test_renderer.py
└── test_semantic_mapping.py
```

## Usage

* Flags

  ```python
  preprocess = False # True if regenerating the matrix for mask and ground truth. Those files will speed up testing.
  latex_mode = True # True if generate latex code of tabels
  visualize = True # True if visualizing global maps and ground truth
  verbose = False # True if print evaluation results for every image False if print final average result
  ```

* Testing

  Please run the following command to output testing results. For the first time run, you need to set `preprocess` to `True` so 
  that the script can generate an integer mask for ground truth map and integer matrix for mask. After this, it is recommended to set `preprocess` to `False`. 
  In this way, we can speed up the testing process. 

  ```bash
  python3 test_semantic_mapping.py
  ```

* Visualization

  To verify if the maps align, users can visalize two maps. Please pass `visualize=True` to `test.full_test` method. It will plot the ground truth map (left) and generated maps (right) with the same kinds of labels.

  ```bash
  python3 test_semantic_mapping.py -v
  ```

  

