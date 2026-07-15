# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates Go1 robot teleoperation using keyboard to set velocity commands
for a trained RSL-RL policy in an arena with ArUco tags.

.. code-block:: bash

    # Usage
    python go1_challenge/scripts/03-go1_nav.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# Remove AprilTag detection import
# from go1_challenge.isaac_sim.worlds.tags_loc import TAGS_LOC

# add argparse arguments
parser = argparse.ArgumentParser(description="Go1 teleoperation with trained policy and keyboard velocity commands.")
parser.add_argument("--disable_fabric", action="store_true", help="Disable Fabric API and use USD instead.")
parser.add_argument(
    "--policy",
    type=str,
    default="unitree_go1_rough/2025-07-17_18-39-56/exported/policy.pt",
    help="Path to the trained policy file.",
)
parser.add_argument("--visualize_tags", action="store_true", help="Visualize detected AprilTags in a separate window.")
parser.add_argument(
    "--teleop",
    action="store_true",
    help="Use keyboard teleoperation. If false, use NavController for autonomous navigation.",
)
parser.add_argument(
    "--level",
    type=int,
    default=1,
    choices=[1, 2, 3],
    help="Challenge level: 1=flat no obstacles, 2=flat with obstacles, 3=rough with obstacles",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import numpy as np
import torch
import carb
import cv2
import time
import os

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import read_file
from isaacsim.core.utils.viewports import set_camera_view
from isaaclab.devices import Se2Keyboard

from go1_challenge.navigation import NavController

##
# Pre-defined configs
##
from go1_challenge.isaaclab_tasks.go1_locomotion.go1_challenge_env_cfg import Go1ChallengeSceneCfg

PKG_PATH = Path(__file__).parent.parent
DEVICE = "cpu"

# Frequencies for navigation and localization updates
# Main loop runs at 50 Hz
OBS_FREQ = 10  # Frequency for observations updates
VIDEO_FREQ = 10  # Frequency for video capture
NAV_FREQ = 5  # Frequency for action updates


def load_policy_rsl(policy_file: str):
    """Load the trained RSL-RL policy.

    Args:
        policy_file (str): Path to the policy file. Relative to the project root.
    """
    # policy_dir =   # / "logs" / "rsl_rl"  # Adjust path as needed
    policy_path: Path = PKG_PATH / policy_file
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file '{policy_path}' does not exist.")

    print(f"[INFO] Loading policy from {policy_path}")

    file_bytes = read_file(policy_path)
    try:
        policy = torch.jit.load(file_bytes)

    except RuntimeError as e:
        raise RuntimeError(f"Failed to load policy from {policy_path}: {e}. Did you properly export the policy?")

    return policy


def load_gym_env(level: int = 1) -> ManagerBasedRLEnv:
    """Load the Gym environment with specified challenge level."""
    env_cfg = Go1ChallengeSceneCfg(level=level)

    env_cfg.sim.device = DEVICE
    if DEVICE == "cpu":
        env_cfg.sim.use_fabric = False

    env_cfg.episode_length_s = 60.0

    env = ManagerBasedRLEnv(cfg=env_cfg)
    return env


def quit_cb():
    """Dummy callback function executed when the key 'Q' is pressed."""
    print("Quit callback")
    simulation_app.close()


def get_camera_intrinsics(camera_cfg):
    """Extract camera intrinsic parameters from IsaacLab camera configuration."""
    # Camera parameters from CameraCfg
    focal_length = camera_cfg.spawn.focal_length  # in mm
    horizontal_aperture = camera_cfg.spawn.horizontal_aperture  # in mm
    width = camera_cfg.width
    height = camera_cfg.height

    # Convert focal length to pixels
    fx = fy = (focal_length * width) / horizontal_aperture

    # Principal point (assuming centered)
    cx = width / 2.0
    cy = height / 2.0

    # Camera parameters tuple format for NavController: (fx, fy, cx, cy)
    camera_params = (fx, fy, cx, cy)

    return camera_params


def get_camera_data(env) -> dict:
    """Get camera data from the environment."""
    camera = env.scene["camera"]

    if camera is not None:
        rgb_data = camera.data.output["rgb"][0, ..., :3].cpu().numpy()
        distance_data = camera.data.output["distance_to_image_plane"][0].cpu().numpy()
        return {"camera_rgb": rgb_data, "camera_distance": distance_data}

    else:
        return None


def get_observation_dict(obs: dict) -> dict:
    """Convert observation tensor to a dictionary for easier access."""
    obs_dict = {
        "base_ang_vel": obs["policy"][:, 0:3].cpu().numpy().flatten(),
        "projected_gravity": obs["policy"][:, 3:6].cpu().numpy().flatten(),
        "velocity_commands": obs["policy"][:, 6:9].cpu().numpy().flatten(),
        "joint_pos": obs["policy"][:, 9:21].cpu().numpy().flatten(),
        "joint_vel": obs["policy"][:, 21:33].cpu().numpy().flatten(),
        "actions": obs["policy"][:, 33:45].cpu().numpy().flatten(),
    }
    return obs_dict


def get_goal_position(env) -> np.ndarray:
    """Get the current goal position from the environment."""
    try:
        goal_asset = env.scene["goal"]
        goal_pos = goal_asset.get_world_poses()[0][0].cpu().numpy()  # Get position for env 0
        return goal_pos[:2]  # Return only x, y coordinates

    except (KeyError, AttributeError):
        print("[WARNING] Goal asset not found in scene")
        return np.array([2.0, 2.0])  # Default goal position


def get_robot_position(env) -> tuple:
    """Get the current robot position and orientation from the environment."""
    robot_asset = env.scene["robot"]
    robot_pos = robot_asset.data.root_pos_w[0].cpu().numpy()
    robot_quat = robot_asset.data.root_quat_w[0].cpu().numpy()  # [w, x, y, z]

    # Convert quaternion to yaw angle
    def quat_to_yaw(q):
        """Convert quaternion (w, x, y, z) to yaw angle in radians."""
        w, x, y, z = q
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return yaw

    yaw = quat_to_yaw(robot_quat)

    robot_pose = [robot_pos[0], robot_pos[1], yaw]

    return robot_pose  # Return (x, y, yaw) position and yaw


def main():
    """Main teleoperation loop."""

    # Setup environment with specified level
    env = load_gym_env(level=args_cli.level)

    # --- Setup keyboard interface
    sensitivity_lin = 1.0
    sensitivity_ang = 1.0

    teleop_interface = Se2Keyboard(v_x_sensitivity=sensitivity_lin, v_y_sensitivity=sensitivity_lin, omega_z_sensitivity=sensitivity_ang)

    teleop_interface.add_callback("R", env.reset)
    teleop_interface.add_callback("Q", quit_cb)

    print(teleop_interface)

    if args_cli.teleop:
        print("\n[INFO] Teleoperation mode enabled. Use Arrows+ZX to control the robot.")

    else:
        print("\n[INFO] Autonomous navigation mode enabled. NavController will control the robot.")

    # --- Load trained policy
    policy_file = args_cli.policy
    policy = load_policy_rsl(policy_file)
    policy = policy.to(env.device).eval()

    set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

    # --- Reset environment
    if teleop_interface is not None:
        teleop_interface.reset()

    print("[DEBUG] Calling env.reset()...")
    obs, _ = env.reset()

    count = 0
    nav_update_count = 1 / (NAV_FREQ * env.step_dt)
    video_update_count = 1 / (VIDEO_FREQ * env.step_dt)
    obs_update_count = 1 / (OBS_FREQ * env.step_dt)

    # --- Initialize NavController, for autonmous navigation
    camera_cfg = env.cfg.scene.camera
    camera_params = get_camera_intrinsics(camera_cfg)
    nav_controller = NavController(
        camera_params=camera_params,
        tag_size=0.16,  # Length of the black square in meters
        tag_family="tag36h11",
    )
    last_nav_command = np.zeros(3)  # Initialize last command

    print(f"[INFO] Challenge Level {args_cli.level} loaded")
    print(f"[INFO] NavController initialized:")
    print(f"  Camera params: {camera_params}")
    print(f"  Tag size: 0.16m")
    print(f"  Tag family: tag36h11")

    # --- Main loop (50 Hz)
    while simulation_app.is_running():
        with torch.inference_mode():
            # --- Nav - send observations
            if count % obs_update_count == 0:
                obs_dict = get_observation_dict(obs)

                # Get goal and robot positions
                goal_pos = get_goal_position(env)
                robot_pose = get_robot_position(env)

                # Add goal and robot positions to observations
                nav_observations = {
                    **obs_dict,
                    "goal_position": goal_pos,
                    "robot_pose": robot_pose,
                }

                nav_controller.update(nav_observations)

            # --- Nav - send video frames
            if count % video_update_count == 0:
                # --- Get camera data
                camera_data_dict = get_camera_data(env)

                nav_controller.update(camera_data_dict)

            # --- Nav - get nav command
            if count % nav_update_count == 0 and not args_cli.teleop:
                last_nav_command = nav_controller.get_command()
                # print(f"[NAV] NavController command: {last_nav_command}")

            # # Debug info
            # goal_distance = np.linalg.norm(goal_pos - robot_pose[:2])
            # print(
            #     f"[DEBUG] Robot: ({robot_pose[0]:.2f}, {robot_pose[1]:.2f}, {np.degrees(robot_pose[2]):.1f}°), ",
            #     f"Goal: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f}), ",
            #     f"Dist: {goal_distance:.2f}m ",
            # )

            # --- Step simulation
            # Get velocity command based on mode
            if args_cli.teleop and teleop_interface is not None:
                command = teleop_interface.advance()
                # print(f"[TELEOP] Keyboard command: {command}")

            # Use NavController commands for autonomous navigation
            else:
                command = last_nav_command

            # Update policy observation with velocity command
            obs["policy"][:, 6:9] = torch.tensor(command)

            # Policy inference
            action = policy(obs["policy"])

            # Step environment
            obs, _, _, _, _ = env.step(action)

            count += 1

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
