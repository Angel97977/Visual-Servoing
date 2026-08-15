# Visual Servoing

This repository stores the custom ROS 2 visual servoing workspace for the xArm camera project.

## Contents
- `mi_robot_vision/` : custom vision and robot movement scripts
- `xarm_ros2/` : upstream xArm ROS 2 packages (already a separate Git repository)

## Typical setup

```bash
cd ~/xarm_cam_ws/src
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Launch the camera setup

```bash
ros2 launch mi_robot_vision xarm_cam.launch.py
```

## Notes
- The `xarm_ros2` folder is a cloned upstream project and already has its own Git remote.
- This repository is intended as a backup for the custom visual servoing workspace and project notes.
- Keep this repo updated with your custom scripts, launch files, and documentation.
