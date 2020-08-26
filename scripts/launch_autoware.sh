#!/bin/bash
# Workspace for AVL
MY_PATH="$HOME/codebase/Autoware_v1.8/ros"

# Output MY_PATH to user
echo "Starting AVL workspace with path "$MY_PATH

# Move to MY_PATH, then we will open 4 terminals
# 1. Runs runtime_manager
# 2. Runs run_proc_manager
# 3. Run roscore
# 4. Open a free terminal for you to play around with.
gnome-terminal --working-directory=$MY_PATH \
--tab --command "bash -c \"source /opt/ros/kinetic/setup.bash; echo sourced window; exec bash\"" \
--tab --command "bash -c \"source /opt/ros/kinetic/setup.bash; echo sourced window; source ./devel/setup.bash; rosrun runtime_manager runtime_manager_dialog.py; exec bash\"" \
--tab --command "bash -c \"source /opt/ros/kinetic/setup.bash; gksudo --message 'Please input password for launching process manager' -- $MY_PATH/run_proc_manager; exec bash\"" \
--tab --command "bash -c \"source /opt/ros/kinetic/setup.bash; echo sourced window; source ./devel/setup.bash; roscore; exec bash\""
