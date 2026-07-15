"""
Definition of the Go1 Challenge environment configuration for IsaacSim. This environment is designed to test the Go1
locomotion policy in the arena with obstacles and ArUco tags.
"""

from pathlib import Path
import torch
import os
import numpy as np


import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObjectCfg

from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg, ManagerBasedRLEnv
import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen

from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, check_file_path, read_file
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab.sensors import CameraCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns

from isaaclab.devices import Se2Keyboard

from pxr import UsdGeom, Gf, UsdPhysics, Sdf
import omni.usd

from isaacsim.core.experimental.prims import XformPrim

##
# Pre-defined configs
##
from go1_challenge.isaaclab_tasks.go1_locomotion.go1_locomotion_env_cfg import Go1LocomotionEnvCfg_PLAY

##
# Scene
##
# import go1_challenge.arena_assets

ARENA_ASSETS_DIR = Path(__file__).parent.parent.parent / "arena_assets"


@configclass
class Go1ChallengeSceneCfg(Go1LocomotionEnvCfg_PLAY):
    """Design the scene with Go1 robot and camera."""

    # Class attribute to store the level
    level: int = 1

    def __post_init__(self):
        """Post initialization to add dynamically generated components."""
        super().__post_init__()

        # --- Arena
        # Walls and Tags
        arena_usd_path = ARENA_ASSETS_DIR / "arena_5x5.usd"
        if not arena_usd_path.exists():
            raise FileNotFoundError(f"Arena USD file not found: {arena_usd_path}")

        self.scene.arena_walls = AssetBaseCfg(
            prim_path="/World/arena_structure",
            spawn=sim_utils.UsdFileCfg(
                usd_path=arena_usd_path.as_posix(),
                # Only spawn the wall/tag structures, not the ground plane
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            collision_group=-1,
        )

        # --- Ground
        if self.level in [1, 2]:
            # Level 1 & 2: Flat arena with AprilTags
            arena_terrain_cfg = TerrainGeneratorCfg(
                size=(2.5, 2.5),  # Slightly larger than arena to cover edges
                border_width=0.1,
                num_rows=2,
                num_cols=2,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                slope_threshold=0.75,
                use_cache=False,
                sub_terrains={
                    "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
                },
            )

        else:
            # Level 3: Arena with rough terrain overlay
            arena_terrain_cfg = TerrainGeneratorCfg(
                size=(2.5, 2.5),  # Slightly larger than arena to cover edges
                border_width=0.02,
                num_rows=2,
                num_cols=2,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                slope_threshold=0.75,
                use_cache=False,
                sub_terrains={
                    "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                        proportion=0.5, grid_width=0.45, grid_height_range=(0.025, 0.1), platform_width=2.0
                    ),
                    "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                        proportion=0.5, noise_range=(0.01, 0.04), noise_step=0.01, border_width=0.25
                    ),
                    # "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.0),
                },
            )

        # Use multi-terrain approach: arena base + rough heightfield
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=arena_terrain_cfg,
            max_init_terrain_level=0,  # Single terrain level
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )

        # -- Goal
        self.scene.goal = AssetBaseCfg(
            prim_path="/World/goal",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(1, 1, 0.0)),
            spawn=sim_utils.UsdFileCfg(
                usd_path=(ARENA_ASSETS_DIR / "assets" / "goal.usd").as_posix(),
                scale=(1, 1, 1),
                # variants={"Tag": "_15"},
            ),
            collision_group=-1,
        )

        # Add goal randomization event
        self.events.randomize_goal = EventTerm(
            func=randomize_goal_position,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("goal"),
                "robot_cfg": SceneEntityCfg("robot"),
                "arena_size": 5.0,
                "wall_buffer": 1.0,
                "min_robot_dist": 3.0,  # Minimum distance from robot spawn
                "max_attempts": 100,
            },
        )

        # -- Dynamic obstacles (Level 2 & 3 only)
        if self.level >= 2:
            self.scene.obstacle_1 = AssetBaseCfg(
                prim_path="/World/obstacles/obstacle_1",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(1.0, 1.0, 0.0)),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=(ARENA_ASSETS_DIR / "assets" / "obstacle.usd").as_posix(),
                    scale=(1.0, 1.0, 1),
                    variants={"Tag": "_15"},
                ),
                collision_group=-1,
            )

            self.scene.obstacle_2 = AssetBaseCfg(
                prim_path="/World/obstacles/obstacle_2",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.0, -1.0, 0.0)),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=(ARENA_ASSETS_DIR / "assets" / "obstacle.usd").as_posix(),
                    scale=(1.0, 1.0, 1),
                    variants={"Tag": "_16"},
                ),
                collision_group=-1,
            )

            self.scene.obstacle_3 = AssetBaseCfg(
                prim_path="/World/obstacles/obstacle_3",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.0, -1.0, 0.0)),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=(ARENA_ASSETS_DIR / "assets" / "obstacle.usd").as_posix(),
                    scale=(1.0, 1.0, 1),
                    variants={"Tag": "_16"},
                ),
                collision_group=-1,
            )

            # Add obstacle randomization event
            self.events.randomize_obstacle_positions = EventTerm(
                func=randomize_obstacle_positions,
                mode="reset",
                params={
                    "asset_cfgs": [
                        SceneEntityCfg("obstacle_1"),
                        SceneEntityCfg("obstacle_2"),
                        SceneEntityCfg("obstacle_3"),
                    ],
                    "avoid_assets_cfg": [
                        {"asset_cfg": SceneEntityCfg("robot"), "buffer_distance": 1.0},  # Keep 2m from robot
                        {"asset_cfg": SceneEntityCfg("goal"), "buffer_distance": 2.0},  # Keep 2m from goal
                    ],
                    "arena_size": 5.0,
                    "wall_buffer": 0.8,
                    "obstacle_buffer": 1.2,  # Distance between obstacles
                },
            )

        # -- Camera (all levels)
        self.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/trunk/front_cam",
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.25, 0.0, 0.08), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        )

        # -- Robot spawn configuration
        self.scene.robot.init_state.pos = (0, 0, 0.4)
        self.scene.num_envs = 1

        # -- MDP settings
        self.episode_length_s = 60.0
        self.curriculum = None
        self.observations.policy.velocity_commands = ObsTerm(func=constant_commands)

        # Update robot spawn to be in bottom left corner safe zone.
        # NOTE: The (0, 0) for this not the center of arena, but the center of the terrain 0.
        # It corresponds to (-1.25, -1.25) in world coordinates.
        self.events.reset_base.params["pose_range"] = {"x": (-0.8, 3.3), "y": (-0.8, 3.3), "yaw": (-3.14, 3.14)}

        print(f"[INFO] Go1 Challenge Level {self.level} configured:")
        if self.level == 1:
            print("  - Flat arena with AprilTags")
            print("  - No obstacles")

        elif self.level == 2:
            print("  - Flat arena with AprilTags")
            print("  - 3 dynamic obstacles")

        elif self.level == 3:
            print("  - Arena walls and AprilTags")
            print("  - Rough terrain heightfield")
            print("  - 3 dynamic obstacles")


##
# MDP settings
##


def constant_commands(env: ManagerBasedEnv) -> torch.Tensor:
    """The generated command from the command generator."""
    return torch.tensor([[1, 0, 0]], device=env.device).repeat(env.num_envs, 1)


def randomize_obstacle_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: SceneEntityCfg,
    avoid_assets_cfg: list,
    arena_size,
    wall_buffer,
    obstacle_buffer,
):
    """Randomly position obstacles in the arena avoiding walls, other obstacles, and specified assets with custom distances."""

    stage = omni.usd.get_context().get_stage()

    # Define boundaries
    min_coord = -arena_size / 2 + wall_buffer
    max_coord = arena_size / 2 - wall_buffer

    # --- Get positions of assets to avoid with their custom buffer distances
    avoid_positions_with_buffers = []
    for avoid_config in avoid_assets_cfg:
        try:
            # Handle both old format (SceneEntityCfg) and new format (dict with asset_cfg and buffer_distance)
            if isinstance(avoid_config, dict):
                avoid_asset_cfg = avoid_config["asset_cfg"]
                buffer_distance = avoid_config["buffer_distance"]
            else:
                # Backward compatibility with old format
                avoid_asset_cfg = avoid_config
                buffer_distance = obstacle_buffer  # Use default obstacle buffer

            avoid_asset = env.scene[avoid_asset_cfg.name]

            if hasattr(avoid_asset, "data") and hasattr(avoid_asset.data, "root_pos_w"):
                # Articulation asset (robot)
                avoid_pos = avoid_asset.data.root_pos_w[env_ids].cpu().numpy()

            else:
                # Rigid object asset (goal, other objects)
                avoid_pos_tensor, _ = avoid_asset.get_world_poses()
                avoid_pos = avoid_pos_tensor[env_ids].cpu().numpy()

            # Store positions with custom buffer distances for each environment
            for i, env_id in enumerate(env_ids):
                pos = avoid_pos[i] if len(avoid_pos.shape) > 1 else avoid_pos
                avoid_positions_with_buffers.append(
                    {
                        "position": (pos[0], pos[1]),  # x, y coordinates
                        "buffer_distance": buffer_distance,
                        "name": avoid_asset_cfg.name,
                    }
                )
                # print(
                #     f"[DEBUG] Avoiding {avoid_asset_cfg.name} at ({pos[0]:.2f}, {pos[1]:.2f}) with {buffer_distance:.1f}m buffer"
                # )

        except (KeyError, AttributeError) as e:
            print(
                f"[WARNING] Could not get position for asset {avoid_asset_cfg.name if isinstance(avoid_config, dict) else avoid_config.name}: {e}"
            )
            continue

    # Store positions of placed obstacles to avoid overlaps
    placed_obstacle_positions = []

    # --- Find valid positions for each obstacle
    for obstacle_idx, asset_cfg in enumerate(asset_cfgs):
        max_attempts = 500
        valid_position_found = False

        for attempt in range(max_attempts):
            # Random position within arena bounds
            x = np.random.uniform(min_coord, max_coord)
            y = np.random.uniform(min_coord, max_coord)

            # Check distance from assets to avoid with their custom buffer distances
            valid_position = True
            for avoid_info in avoid_positions_with_buffers:
                avoid_x, avoid_y = avoid_info["position"]
                required_distance = avoid_info["buffer_distance"]
                asset_name = avoid_info["name"]

                distance_to_avoid = np.sqrt((x - avoid_x) ** 2 + (y - avoid_y) ** 2)
                if distance_to_avoid < required_distance:
                    valid_position = False
                    # print(f"[DEBUG] Position ({x:.2f}, {y:.2f}) too close to {asset_name}: {distance_to_avoid:.2f}m < {required_distance:.2f}m")
                    break

            if not valid_position:
                continue

            # Check distance from other already placed obstacles
            for prev_x, prev_y in placed_obstacle_positions:
                distance_to_obstacle = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                if distance_to_obstacle < obstacle_buffer:
                    valid_position = False
                    break

            if valid_position:
                placed_obstacle_positions.append((x, y))
                valid_position_found = True
                # print(f"[DEBUG] Obstacle {asset_cfg.name} placed at ({x:.2f}, {y:.2f})")
                break

        if not valid_position_found:
            # Fallback: place obstacle in a safe position using polar coordinates
            # Distribute obstacles around the arena perimeter
            angle = obstacle_idx * 2 * np.pi / len(asset_cfgs)
            fallback_radius = (arena_size / 2) - wall_buffer - 0.5  # Safe distance from walls
            x = fallback_radius * np.cos(angle)
            y = fallback_radius * np.sin(angle)

            # Ensure fallback position is still within bounds
            x = np.clip(x, min_coord, max_coord)
            y = np.clip(y, min_coord, max_coord)

            placed_obstacle_positions.append((x, y))
            print(
                f"[WARNING] Could not find valid position for {asset_cfg.name} after {max_attempts} attempts. Using fallback at ({x:.2f}, {y:.2f})"
            )

    # --- Apply positions to obstacle assets
    for i, asset_cfg in enumerate(asset_cfgs):
        x, y = placed_obstacle_positions[i]
        z = 0.0  # Height for all obstacles

        # Get asset from scene
        asset: XformPrim = env.scene[asset_cfg.name]
        prim_paths = asset.prim_paths

        for _, env_id in enumerate(env_ids):
            prim_path = prim_paths[env_id]

            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            # Set the position
            translate_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:translate")
            translate_spec.default = Gf.Vec3f(*[x, y, z])

        # print(f"[INFO] Obstacle {asset_cfg.name} positioned at ({x:.2f}, {y:.2f})")


def randomize_goal_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    arena_size: float,
    wall_buffer: float,
    min_robot_dist: float,
    max_attempts: int = 100,
):
    """Randomly position the goal in the arena with minimum distance from robot spawn."""

    stage = omni.usd.get_context().get_stage()

    # Define arena boundaries
    min_coord = -arena_size / 2 + wall_buffer
    max_coord = arena_size / 2 - wall_buffer

    # Get robot asset and its CURRENT position (after reset_base has been called)
    robot_asset = env.scene[robot_cfg.name]

    # # Force update robot data to ensure we get the latest position after reset
    # robot_asset.update_world_poses()
    robot_positions = robot_asset.data.root_pos_w[env_ids].cpu().numpy()

    # Get goal asset
    goal_asset: XFormPrim = env.scene[asset_cfg.name]

    goal_prim_paths = goal_asset.prim_paths

    for i, env_id in enumerate(env_ids):
        robot_pos = robot_positions[i]
        robot_x, robot_y = robot_pos[0], robot_pos[1]

        # print(f"[DEBUG] Robot {env_id} reset position: ({robot_x:.2f}, {robot_y:.2f})")

        # Find valid goal position
        valid_position_found = False
        distance_to_robot = 0.0

        for attempt in range(max_attempts):
            # Random position within arena bounds
            goal_x = np.random.uniform(min_coord, max_coord)
            goal_y = np.random.uniform(min_coord, max_coord)

            # Check minimum distance from robot
            distance_to_robot = np.sqrt((goal_x - robot_x) ** 2 + (goal_y - robot_y) ** 2)

            if distance_to_robot >= min_robot_dist:
                valid_position_found = True
                break

        if not valid_position_found:
            # Fallback: place goal at opposite corner from robot
            if robot_x > 0:
                goal_x = min_coord + 0.5
            else:
                goal_x = max_coord - 0.5

            if robot_y > 0:
                goal_y = min_coord + 0.5
            else:
                goal_y = max_coord - 0.5

            distance_to_robot = np.sqrt((goal_x - robot_x) ** 2 + (goal_y - robot_y) ** 2)
            print(f"[WARNING] Could not find valid goal position after {max_attempts} attempts. Using fallback.")

        # Apply position to goal asset
        goal_z = 0.0  # Ground level
        prim_path = goal_prim_paths[env_id]

        prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)
        translate_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:translate")
        translate_spec.default = Gf.Vec3f(goal_x, goal_y, goal_z)

        print(f"[INFO] Goal {env_id} positioned at ({goal_x:.2f}, {goal_y:.2f}) - {distance_to_robot:.2f}m from robot")
