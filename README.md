# Probabilistic Semantic Mapping for Autonomous Driving in Urban Environment

This is the repository for our submission to [Special Issue "Advanced Sensing Techniques for Autonomous Vehicles and Advanced Driver Assistance Systems (ADAS)"](https://www.mdpi.com/si/sensors/autonomous_vehicles_ADAS).

The source code freeze at our submission can be found in branch [hrnet_sensors_submit](https://github.com/AutonomousVehicleLaboratory/semantic_mapping_v2/tree/hrnet_sensors_submit) and [deeplab_sensors_submit](https://github.com/AutonomousVehicleLaboratory/semantic_mapping_v2/tree/deeplab_sensors_submit).

Our mapping results for different ablation studies and groundtruth labels are available in [Google Drive](https://drive.google.com/drive/folders/1eMRNIizjmEStxS0i_ZaKjc9z9hFkRSqN?usp=sharing). Additonal mapping results are also shared. 


TODO:
1. [x] open-source code
2. [ ] refactor code
3. [ ] make data availble ([ ] anonymized rosbag and [x] ground truth)
4. [ ] user guide with docker
5. [x] add sample results



The below guide is the initial version and will be updated soon.

## Introduction

We will fuse the LiDAR point cloud with the semantic segmented 2D camera image together and create a bird's-eye-view semantic map of the environment. We are developing this repository to adapt new sensor suits and new software architectures. 

## Setup

Refer to this documentation of how to [setup](doc/setup.md) the development environment. 

## Run Experiment

There are still some none trivial step for you to run our experiment presented in the paper, and that is why we are here to guide you. 

To run the whole map generation, you should first set the `Start Time` in the Simulation Tab of the Autoware to `410`.  Then create an configuration file, and have the following keys set to

```yaml
TEST_END_TIME: 1581541450
MAPPING:
  BOUNDARY: [[0, 1000], [0, 1400]]
  RESOLUTION: 0.2
```

To run a small region of the map for testing, you should first set the `Start Time` to `390`. Then use the following configuration 

```yaml
TEST_END_TIME: 1581541270 # Which is about 20s after 390s. 
MAPPING:
  BOUNDARY: [[100, 300], [800, 1000]]
  RESOLUTION: 0.2		# This one doesn't really matter
```

Then pass these configuration into the roslaunch as shown in Step 4 and click the `Play ` in the Autoware, now you should be good to go!

## Reproducibility

To reproduce our result, you need to 

1. Slow down the replay speed of the rosbag. As the image node will drop package if it is legging behind, the drop of package is uncontrollable and can happen at anytime. Therefore we need to slow down the replay speed to ensure that all the packages are processed. We recommend using `0.1` replay rate. 
2. Set the input image scale to 1.0. The nearest neighbor interpolation from the `cv2` may cause some misalignments and we want to disable that. 
3. Fix the random seed. Set the `RNG_SEED` value in the configuration file to a non zero value so that we can disable any randomness. 

## Developer Note

You can install rospy via pip by (Reference is [here](https://answers.ros.org/question/343260/install-of-the-rospyrosbag-python-libraries-standalone-not-on-ubuntu/))

```
pip install --extra-index-url https://rospypi.github.io/simple/ rospy
```

We have provided a script `install_ros_python_packages.sh` in the script folder. You can run it by 

```shell
bash ./install_ros_python_packages.sh
```

### Create ROS node

In order for your Python script to be detectable by the `catkin_make`, you have to make it `executable`. Actually, `catkin_make` will collect all the executable files in your source code and make them visible in `rosrun/roslaunch`. 



## Credits

Author: David, Henry, Qinru, Hao. 