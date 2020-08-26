# Initialize the development environment (You should only call this function once)

# ==========================================================================
# Setup
# ==========================================================================

# ROS will report error as the old key has been revoked.
# https://answers.ros.org/question/325039/apt-update-fails-cannot-install-pkgs-key-not-working/
sudo apt update

# exit when any command fails
# Copied from https://intoli.com/blog/exit-on-errors-in-bash-scripts/
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

function git_clone_if_not_exist() {
  LOCALREPO_VC_DIR=$1
  REPOSRC=$2
  [ -d $LOCALREPO_VC_DIR ] || git clone $REPOSRC $LOCALREPO
}

# ==========================================================================
# Main
# ==========================================================================
# Build the Autoware
mkdir -p ~/codebase
cd ~/codebase

git_clone_if_not_exist Autoware_v1.8 https://github.com/AutonomousVehicleLaboratory/Autoware_v1.8.git

# We need to source the ros environment in order to build the Autoware
source /opt/ros/kinetic/setup.bash
cd Autoware_v1.8/ros
./catkin_make_release

# Build ROS package
mkdir -p ~/codebase/ros_workspace/src
cd ~/codebase/ros_workspace/src
[ -f "CMakeLists.txt" ] ||catkin_init_workspace

git_clone_if_not_exist map_reduction https://github.com/AutonomousVehicleLaboratory/map_reduction.git
git_clone_if_not_exist vision_semantic_segmentation https://github.com/AutonomousVehicleLaboratory/vision_semantic_segmentation.git

# Build the code
cd ..
catkin_make

# Install independence
sudo apt install gnome-terminal -y

# Install python package
cd ~/codebase/ros_workspace/src
pip install numpy --upgrade --user
pip install -r vision_semantic_segmentation/requirements.txt --user
