#!/bin/bash
# A detail explanation of whether we should use #!/bin/bash
# The shebang only really matters if (a) you need to run in something that isn't just a shell, like python or perl, or (b) you don't use the bash shell (ie you use zsh) but you need to run something which requires being run in bash.
# Reference: https://stackoverflow.com/questions/8967902/why-do-you-need-to-put-bin-bash-at-the-beginning-of-a-script-file#:~:text=Bash%20has%20evolved%20over%20the,that%20follow%20in%20the%20script.

# This script will install all the rospy package needed for ros python development
pip install --extra-index-url https://rospypi.github.io/simple/ \
  rospy \
  cv-bridge \
  geometry_msgs \
  sensor_msgs \
  tf
