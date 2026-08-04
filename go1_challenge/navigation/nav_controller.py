"""
Navigation Controller base class for Go1 Challenge.

This class handles robot localization and navigation command generation.
@MRSS26: You have to implement the core methods to complete the navigation system.
"""

import numpy as np
import torch
import cv2
import time
import os
from typing import Any

from pyapriltags import Detector

# Position of the tags in the arena
TAG_POSITIONS = {
    0: [1.25, 2.4, 0.5],
    1: [-1.25, 2.4, 0.5],
    2: [-2.4, 1.25, 0.5],
    3: [-2.4, -1.25, 0.5],
    4: [2.4, 1.25, 0.5],
    5: [2.4, -1.25, 0.5],
    6: [1.25, -2.4, 0.5],
    7: [-1.25, -2.4, 0.5],
}  # Tags that are not here are obstacles, not landmarks


class NavController:
    """
    Base Navigation Controller for Go1 robot navigation with AprilTag localization.

    The controller receives observations at regular intervals and maintains internal
    state for robot pose estimation and navigation planning.
    """

    def __init__(
        self, camera_params: tuple[float, float, float, float], tag_size: float = 0.16, tag_family: str = "tag36h11"
    ):
        """
        Initialize the navigation controller.

        Args:
            camera_params: Camera intrinsics (fx, fy, cx, cy) for AprilTag detection
            tag_size: Physical size of AprilTags in meters (length of black square)
            tag_family: AprilTag family name (e.g., "tag36h11")

        Students should implement their state initialization here, including:
        - Robot pose estimates (position, orientation)
        - Map or landmark storage
        - Kalman filter/particle filter initialization
        - Any other navigation-related state
        """
        # Camera and AprilTag parameters
        self.camera_params = camera_params
        self.tag_size = tag_size
        self.tag_family = tag_family

        # AprilTag detector
        self.at_detector = Detector(
            families=tag_family,
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

        # Example state variables (students can modify/extend):
        self.robot_pose = np.array([0.0, 0.0, 0.0])  # Estimated robot pose in world frame [x, y, yaw]
        self.goal = None  # Current goal position in world frame [x, y]

        self.pose_covariance = np.eye(3) * 0.1  # Pose uncertainty
        self.landmark_map = {}  # Dict to store landmark positions
        self.last_update_time = 0.0

        # @MRSS26: You can add more state variables as needed for your navigation logic
        pass

    def detect_apriltags(self, rgb_image: torch.Tensor, visualize: bool = False) -> dict[int, dict]:
        """
        Detect AprilTags in RGB image and estimate their poses.

        @MRSS26: You can override this method to implement your own detection logic
        or use the raw images for other processing.

        Args:
            rgb_image: RGB image tensor from camera (H, W, 3)
            visualize: Whether to save visualization images to disk

        Returns:
            Dictionary of detected tags with poses in camera frame
            Format: {tag_id: {"pose": {"position": [x,y,z], "rotation_matrix": [[...]]},
                             "distance": float, "confidence": float}}
        """
        try:
            # Convert tensor to numpy and ensure correct format
            if isinstance(rgb_image, torch.Tensor):
                image_np = rgb_image.detach().cpu().numpy()
            else:
                image_np = rgb_image

            # Ensure image is in uint8 format
            if image_np.dtype != np.uint8:
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                else:
                    image_np = image_np.astype(np.uint8)

            # Convert to grayscale for AprilTag detection
            if len(image_np.shape) == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_np

            # Detect tags with pose estimation
            tags = self.at_detector.detect(
                gray, estimate_tag_pose=True, camera_params=self.camera_params, tag_size=self.tag_size
            )

            # --- Process detected tags
            detected_tags = {}

            for tag in tags:
                tag_id = tag.tag_id
                position = tag.pose_t.flatten()  # Translation vector [x, y, z]
                rotation_matrix = tag.pose_R  # 3x3 rotation matrix
                distance = np.linalg.norm(position)

                detected_tags[tag_id] = {
                    "corners": tag.corners.tolist(),
                    "center": tag.center.tolist(),
                    "pose": {
                        "position": position.tolist(),  # [x, y, z] in camera frame
                        "rotation_matrix": rotation_matrix.tolist(),
                    },
                    "distance": float(distance),
                    "confidence": float(tag.decision_margin),
                }

            # Save visualization if requested
            if visualize:
                self._save_visualization_image(
                    image_np, tags, f"tags_{'_'.join(map(str, sorted(detected_tags.keys())))}"
                )

            return detected_tags

        except Exception as e:
            print(f"[ERROR] AprilTag detection failed: {e}")
            return {}

    def _save_visualization_image(self, image_np: np.ndarray, tags: list, prefix: str):
        """Save visualization image with detected tags to disk."""
        try:
            # Convert grayscale to color if needed
            if len(image_np.shape) == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                vis_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            else:
                vis_image = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

            # Draw detected tags
            for tag in tags:
                # Draw tag outline
                corners = tag.corners.astype(int)
                for idx in range(len(corners)):
                    cv2.line(vis_image, tuple(corners[idx - 1]), tuple(corners[idx]), (0, 255, 0), 2)

                # Add tag ID text
                center = tag.center.astype(int)
                cv2.putText(
                    vis_image,
                    f"ID:{tag.tag_id}",
                    org=(center[0] - 20, center[1] - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=(0, 0, 255),
                    thickness=2,
                )

                # Add distance text
                distance = np.linalg.norm(tag.pose_t)
                cv2.putText(
                    vis_image,
                    f"{distance:.2f}m",
                    org=(center[0] - 20, center[1] + 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 0, 0),
                    thickness=1,
                )

            # Save image
            timestamp = int(time.time() * 1000)
            save_dir = "apriltag_detection/"
            os.makedirs(save_dir, exist_ok=True)
            filename = f"{save_dir}/{prefix}_{timestamp}.jpg"

            # Convert RGB to BGR for OpenCV saving
            vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, vis_image_bgr)

            print(f"[INFO] Saved visualization to {filename}")

        except Exception as e:
            print(f"[WARNING] Failed to save visualization: {e}")

    def update(self, observations: dict[str, Any]) -> None:
        """
        Update internal navigation state based on sensor observations.

        This method is called from the main loop (sim or real) when new information is available.

        @MRSS26: It's here where you should implement localization and planning logic here.

        For the sim environemnt, frame HxW is 480x640

        Args:
            observations: Dictionary containing:
                - 'base_ang_vel': Angular velocity [rad/s] (3,) array [roll_rate, pitch_rate, yaw_rate]
                - 'projected_gravity': Gravity vector in body frame (3,) array
                - 'velocity_commands': Current velocity commands (3,) array [vx, vy, wz]
                - 'joint_pos': Joint positions (12,) array
                - 'joint_vel': Joint velocities (12,) array
                - 'actions': Last action taken (12,) array
                - 'goal_position': Goal position in world frame (2,) array [x, y]
                - 'robot_pose': Robot position in world frame (3,) array [x, y, yaw]. This is ONLY to help you debug
                - 'camera_rgb': RGB camera image tensor (H, W, 3)
                - 'camera_distance': Distance to camera in meters (H, W, 1)
                and validate your localization logic. It will not be available on the real robot.

        Note:
            - Not all observations are guaranteed to be present. This can be called with only robot observations
            or only camera observations.
            - rgb_image may be None if camera is not available
        """
        # 1. Process the observations
        # Get RGB image and detect AprilTags
        rgb_image = observations.get("camera_rgb", None)
        detected_tags = {}

        if rgb_image is not None:
            # Students can customize visualization and detection
            detected_tags = self.detect_apriltags(rgb_image, visualize=False)

        # Get other sensor data
        base_ang_vel = observations.get("base_ang_vel", None)
        projected_gravity = observations.get("projected_gravity", None)

        # Get goal and robot positions
        goal_position = observations.get("goal_position", None)

        # Real position, not available on the real robot
        robot_pose_gt = observations.get("robot_pose", None)

        if goal_position is not None:
            self.goal = np.array(goal_position)

        # Example: Print detected tags and goal info for debugging
        if detected_tags:
            print(f"[NavController] Detected tags: {list(detected_tags.keys())}")
            for tag_id, tag_info in detected_tags.items():
                print(f"\t-{tag_id}: Position {tag_info['pose']['position']}, Distance {tag_info['distance']:.2f}m")

        # 2. Localization and planning logic
        # TODO @MRSS26: Implement localization logic here
        self.robot_pose = ...

        pass

    def get_command(self) -> np.ndarray:
        """
        Generate velocity command based on current navigation state.

        @MRSS26: You should implement navigation/path planning logic here. Ideas of observations you could use:
        - Current robot pose estimate (self.robot_pose)
        - Goal position (from latest observations)
        - Detected AprilTags for localization
        - Obstacle avoidance logic

        Returns:
            Array (3, ) with the structure  [lin_vel_x, lin_vel_y, ang_vel_z] in robot body frame.
            All values should be in range [-1, 1] representing normalized velocities.
        """
        # Example placeholder - simple forward motion:
        lin_vel_x = 0.0  # Move forward at reduced speed
        lin_vel_y = 0.0  # No lateral motion
        ang_vel_z = 0.0  # No rotation

        # TODO MRSS26: Implement navigation logic

        # Ensure commands are in valid range
        lin_vel_x = np.clip(lin_vel_x, -1.0, 1.0)
        lin_vel_y = np.clip(lin_vel_y, -1.0, 1.0)
        ang_vel_z = np.clip(ang_vel_z, -1.0, 1.0)

        command = np.array([lin_vel_x, lin_vel_y, ang_vel_z], dtype=np.float32)

        return command

    def reset(self) -> None:
        """
        Reset the navigation controller state.

        Called when the environment/robot is reset.
        """
        # Students implement reset logic here
        self.robot_pose = np.array([0.0, 0.0, 0.0])
        self.pose_covariance = np.eye(3) * 0.1
        self.landmark_map.clear()

        # TODO @MRSS26: Implement reset logic

        pass

    def get_debug_info(self) -> dict[str, Any]:
        """Get debug information for visualization/logging."""

        return {
            "robot_pose": self.robot_pose.tolist(),
            "pose_covariance": self.pose_covariance.tolist(),
            "landmark_count": len(self.landmark_map),
            "landmark_map": self.landmark_map,
            "camera_params": self.camera_params,
            "tag_size": self.tag_size,
            "tag_family": self.tag_family,
        }
