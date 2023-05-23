FROM nvidia/cuda:11.7.0-devel-ubuntu20.04

# Install common 
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y \
    curl \
    wget \
    ca-certificates \
    vim \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    gnupg \
    lsb-release \
    gnupg2 \
    tmux \
 && rm -rf \var/lib/apt/lists/*

# ROS Noetic Installation
RUN sudo apt update && sudo apt install -y locales
RUN sudo locale-gen en_US en_US.UTF-8
RUN sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
RUN export LANG=en_US

RUN DEBIAN_FRONTEND=noninteractive TZ=America/US apt-get -y install tzdata
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y keyboard-configuration

RUN sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

RUN sudo apt-get update
RUN DEBIAN_FRONTEND=noninteractive sudo apt install -y ros-noetic-desktop
RUN sudo apt install -y ros-noetic-ros-base
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
# Install Rosdep
RUN sudo apt install -y python3-rosdep
RUN sudo apt-get update

RUN rosdep init
RUN rosdep update

RUN sudo apt update
RUN sudo apt install ros-noetic-compressed-image-transport
RUN sudo apt install ros-noetic-image-transport-plugins -y
# Final update
RUN sudo apt-get update
