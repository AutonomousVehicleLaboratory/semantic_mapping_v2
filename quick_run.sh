#!/bin/bash

# Build docker containers
sh ./docker/base-cuda
sh ./docker/semantic-mapping

# Run docker container, bind the semantic_mapping_v2 to the src of home/rosws
docker run --name quick-mapping -it --gpus all -v "$(pwd):/home/rosws/src/semantic_mapping_v2:rw" --rm vision-semantic-segmentation:noetic /bin/bash -c "

# Add map_reduction
cd /home/rosws/src;
ls;
git clone https://github.com/AutonomousVehicleLaboratory/map_reduction;

# Initialize Catkin workspace
cd /home/rosws/;
catkin_init_workspace;

# Make workspace
catkin_make;
"