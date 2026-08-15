import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declaramos la IP del robot (usando la tuya por defecto para ahorrar tiempo)
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip', 
        default_value='192.168.1.167', 
        description='IP of the xArm Lite 6'
    )

    # 1. Incluir el Launch del xArm Lite 6
    xarm_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('xarm_moveit_config'),
                'launch',
                'lite6_moveit_realmove.launch.py'
            ])
        ]),
        launch_arguments={'robot_ip': LaunchConfiguration('robot_ip')}.items(),
    )

    # 2. Nodo de la cámara USB
    usb_cam_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam',
        output='screen',
        parameters=[{
            'video_device': '/dev/video0',
            'framerate': 30.0,
            'image_width': 640,
            'image_height': 480
        }]
    )

    return LaunchDescription([
        robot_ip_arg,
        xarm_launch,
        usb_cam_node
    ])
