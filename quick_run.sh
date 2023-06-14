#!/bin/bash

# Build docker containers
sh ./docker/base-cuda
sh ./docker/semantic-mapping

# Run docker container, bind the semantic_mapping_v2 to the src of home/rosws
docker run --name quick-mapping -it --gpus all -v "$(pwd):/home/rosws/src/semantic_mapping_v2:rw" --rm vision-semantic-segmentation:noetic /bin/bash -c "

# Add map_reduction and catkin
cd /home/rosws/src;
git clone https://github.com/AutonomousVehicleLaboratory/map_reduction;
git clone https://github.com/ros/catkin.git;

# Install catkin_tools
sudo apt-get update
sudo apt-get install python3-osrf-pycommon python3-catkin-tools

# Initialize Catkin workspace
cd /home/rosws/;
catkin init;

# Make workspace
catkin build;
"