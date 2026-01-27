# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import imp
import torch
import torch.distributions as D
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import Unbounded, Composite, DiscreteTensorSpec, BinaryDiscreteTensorSpec

import isaacsim.core.utils.prims as prim_utils
import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import euler_to_quaternion, quat_rotate, quat_rotate_inverse
from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import ArticulationView, RigidPrimView

from omni_drones.robots import ASSET_PATH

from pxr import UsdPhysics

# Debug visualization
try:
    from isaacsim.util.debug_draw import _debug_draw
    DEBUG_DRAW_AVAILABLE = True
except ImportError:
    DEBUG_DRAW_AVAILABLE = False
    _debug_draw = None

class DroneRaceEnv(IsaacEnv):
    r"""
    A drone racing task where the agent must navigate through a sequence of gates
    in a racing track. The gates are arranged in a track pattern and the agent
    must pass through them in order.

    ## Observation

    - `drone_state` (16 + num_rotors): The basic information of the drone (except its position),
      containing its rotation (in quaternion), velocities (linear and angular),
      heading and up vectors, and the current throttle.
    - `next_gate_rpos` (3): The relative position of the next gate to the drone in the drone's local frame.
    - `next_gate_rpos_world` (3): The relative position of the next gate to the drone in world frame.
    - `next_gate_orientation` (4): The orientation (quaternion) of the next gate.
    - `gate_progress` (1): Progress through the track (current_gate_index / total_gates).
    - `time_encoding` (optional): The time encoding, which is a 4-dimensional
      vector encoding the current progress of the episode.

    ## Reward

    - `progress`: Reward for making progress toward the next gate.
    - `gate_passage`: Large bonus reward for successfully passing through a gate.
    - `track_progress`: Reward for overall progress through the track.
    - `up`: Reward for maintaining an upright orientation.
    - `effort`: Reward computed from the effort of the drone to optimize the
      energy consumption.
    - `spin`: Reward computed from the spin of the drone to discourage spinning.

    The total reward is computed as follows:

    ```{math}
        r = r_\text{progress} + r_\text{gate_passage} + r_\text{track_progress} + (r_\text{up} + r_\text{spin}) + r_\text{effort}
    ```

    ## Episode End

    The episode ends when the drone gets too close to the ground, crashes,
    or when the maximum episode length is reached. Optionally, the episode can
    end when the drone completes the track.

    ## Config

    | Parameter               | Type  | Default       | Description                                                                                                                                                                                                                             |
    | ----------------------- | ----- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `drone_model`           | str   | "Hummingbird" | Specifies the model of the drone being used in the environment.                                                                                                                                                                         |
    | `track_config`          | dict  | None          | Optional dictionary defining gate positions and orientations. If provided, gates are placed according to this config. Format: `{"1": {"pos": (x, y, z), "yaw": angle}, ...}`. If None, uses circular track.                            |
    | `num_gates`             | int   | 8             | Number of gates in the racing track (only used if `track_config` is None).                                                                                                                                                              |
    | `track_radius`          | float | 5.0           | Radius of the circular track (only used if `track_config` is None).                                                                                                                                                                     |
    | `gate_spacing`          | float | 3.0           | Spacing between gates along the track (only used if `track_config` is None).                                                                                                                                                           |
    | `gate_height`           | float | 2.0           | Height of the gates (only used if `track_config` is None).                                                                                                                                                                              |
    | `gate_scale`            | float | 1.0           | Scale of the gate assets.                                                                                                                                                                                                              |
    | `gate_asset_path`       | str   | None          | Path to the gate USD asset. Defaults to `ASSET_PATH/gate/gate.usd` (isaac_drone_racer style). Can be overridden in config.                                                                                                        |
    | `reward_final_position` | float | 10.0          | Reward for reaching the final position after passing all gates.                                                                                                                                                                      |
    | `pass_threshold`        | float | 0.8           | Distance threshold for passing through a gate.                                                                                                                                                                                        |
    | `reward_progress_scale` | float | 2.0           | Scale for progress reward.                                                                                                                                                                                                             |
    | `reward_gate_passage`   | float | 10.0          | Reward for passing through a gate.                                                                                                                                                                                                     |
    | `reward_effort_weight`  | float | 0.1           | Weight for effort reward.                                                                                                                                                                                                              |
    | `time_encoding`         | bool  | True          | Indicates whether to include time encoding in the observation space.                                                                                                                                                                   |
    | `num_laps`              | int   | 1             | Number of laps the drone must complete before landing near the first gate.                                                                                                                                                             |
    """
    def __init__(self, cfg, headless):
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_progress_scale = cfg.task.reward_progress_scale
        self.reward_gate_passage = cfg.task.reward_gate_passage
        self.reward_final_position = cfg.task.reward_final_position
        self.time_encoding = cfg.task.time_encoding
        self.gate_scale = cfg.task.gate_scale
        self.pass_threshold = cfg.task.pass_threshold
        self.reset_on_collision = cfg.task.get("reset_on_collision", False)
        self.num_laps = cfg.task.get("num_laps", 1)  # Number of laps to complete
        
        # Gate asset path - default to isaac_drone_racer gate asset
        # User can override this in config: gate_asset_path: "path/to/gate.usd"
        # If not specified, defaults to gate/gate.usd (isaac_drone_racer style)
        # Gate asset is in ASSET_PATH/gate/gate.usd
        self.gate_asset_path = cfg.task.get("gate_asset_path", ASSET_PATH + "/gate/gate.usd")
        
        # Track configuration: support both config-based and circular track
        self.track_config = cfg.task.get("track_config", None)
        if self.track_config is not None:
            # Config-based track: gates defined with positions and yaw angles
            self.num_gates = len(self.track_config)
            self.track_type = "config"
            # For config-based tracks, use gate_height from config or default
            self.gate_height = cfg.task.get("gate_height", 2.0)
        else:
            # Circular track (backward compatibility)
            self.num_gates = int(cfg.task.num_gates)
            self.track_radius = cfg.task.track_radius
            self.gate_spacing = cfg.task.gate_spacing
            self.gate_height = cfg.task.gate_height
            self.track_type = "circular"
        
        import traceback
        import sys
        
        print(f"[DroneRaceEnv] Initializing, num_gates={self.num_gates}")
        try:
            super().__init__(cfg, headless)
            print(f"[DroneRaceEnv] super().__init__ completed")
        except Exception as e:
            print("=" * 80)
            print("ERROR: Failed in super().__init__")
            print("=" * 80)
            traceback.print_exc()
            print("=" * 80)
            raise

        try:
            self.drone.initialize()
            print(f"[DroneRaceEnv] drone.initialize() completed")
        except Exception as e:
            print("=" * 80)
            print("ERROR: Failed to initialize drone")
            print("=" * 80)
            traceback.print_exc()
            print("=" * 80)
            raise

        # Track gate progress for each environment
        self.gate_indices = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.gate_passed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # Use a single view with wildcard pattern to access all gates
        try:
            print(f"[DroneRaceEnv] Creating RigidPrimView with pattern='/World/envs/env_*/Gate_*', shape=[{self.num_envs}, {self.num_gates}]")
            self.gates = RigidPrimView(
                "/World/envs/env_*/Gate_*",
                reset_xform_properties=False,
                shape=[self.num_envs, self.num_gates],
                track_contact_forces=self.reset_on_collision
            )
            print(f"[DroneRaceEnv] RigidPrimView created, calling initialize()...")
            self.gates.initialize()
            print(f"[DroneRaceEnv] gates.initialize() completed")
        except Exception as e:
            print("=" * 80)
            print(f"ERROR: Failed to initialize gates view with num_envs={self.num_envs}, num_gates={self.num_gates}")
            print("=" * 80)
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {str(e)}")
            print("\nFull traceback:")
            traceback.print_exc()
            print("=" * 80)
            sys.stderr.write("=" * 80 + "\n")
            sys.stderr.write(f"ERROR: Failed to initialize gates view\n")
            traceback.print_exc(file=sys.stderr)
            sys.stderr.write("=" * 80 + "\n")
            raise  # Re-raise to see the full error
        
        # For collision detection, we can use the same view or access frame specifically
        # Since gates are static, we can use the main gates view for collision if needed
        self.gate_frames = self.gates if self.reset_on_collision else None

        self.init_vels = torch.zeros_like(self.drone.get_velocities())
        self.init_joint_pos = self.drone.get_joint_positions(True)
        self.init_joint_vels = torch.zeros_like(self.drone.get_joint_velocities())

        self.init_pos_dist = D.Uniform(
            torch.tensor([-1.0, -1.0, 1.5], device=self.device),
            torch.tensor([1.0, 1.0, 2.5], device=self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([.2, .2, 0.], device=self.device) * torch.pi
        )

        self.offset_local = torch.tensor([-1.5, 0.0, 0.0], device=self.device)
        self.alpha = 0.8
        
        # Debug visualization: enable via config (default: False)
        self.debug_gate_origins = cfg.task.get("debug_gate_origins", False)
        if self.debug_gate_origins and DEBUG_DRAW_AVAILABLE:
            self.draw = _debug_draw.acquire_debug_draw_interface()
            self.axis_length = cfg.task.get("debug_axis_length", 0.3)  # Length of axis lines
        else:
            self.draw = None

    def _draw_gate_origins(self, gate_world_pos, gate_world_rot, env_idx=0):
        """
        Draw coordinate axes at gate positions to visualize where the gate frame origin is.
        
        Args:
            gate_world_pos: (num_envs, num_gates, 3) tensor of gate positions in world coordinates
            gate_world_rot: (num_envs, num_gates, 4) tensor of gate rotations (quaternions) in world coordinates
            env_idx: Which environment to visualize (default: 0, first environment)
        """
        if self.draw is None:
            return
        
        # Clear previous lines
        self.draw.clear_lines()
        
        # Select one environment to visualize
        gate_pos = gate_world_pos[env_idx]  # (num_gates, 3) - world coordinates
        gate_rot = gate_world_rot[env_idx]  # (num_gates, 4) - world coordinates
        
        # Define axis directions in local frame (gate's local frame)
        # X-axis (red): forward direction
        # Y-axis (green): right direction  
        # Z-axis (blue): up direction
        axis_dirs_local = torch.tensor([
            [self.axis_length, 0.0, 0.0],  # X-axis (red)
            [0.0, self.axis_length, 0.0],  # Y-axis (green)
            [0.0, 0.0, self.axis_length],  # Z-axis (blue)
        ], device=self.device, dtype=torch.float32)  # (3, 3)
        
        # Colors: Red for X, Green for Y, Blue for Z (RGBA)
        axis_colors = [
            (1.0, 0.0, 0.0, 1.0),  # Red for X
            (0.0, 1.0, 0.0, 1.0),  # Green for Y
            (0.0, 0.0, 1.0, 1.0),  # Blue for Z
        ]
        
        # Draw axes for each gate
        for gate_idx in range(gate_pos.shape[0]):
            gate_origin = gate_pos[gate_idx]  # (3,)
            gate_quat = gate_rot[gate_idx]  # (4,)
            
            # Rotate axis directions from gate's local frame to world frame
            # quat_rotate expects (N, 4) and (N, 3), so we need to rotate each axis separately
            # Expand gate_quat to match number of axes (3 axes)
            gate_quat_expanded = gate_quat.unsqueeze(0).expand(3, -1)  # (3, 4)
            axis_dirs_world = quat_rotate(
                gate_quat_expanded,  # (3, 4)
                axis_dirs_local  # (3, 3) - each row is a 3D vector
            )  # (3, 3) - each row is a rotated 3D vector
            
            # Draw each axis
            for axis_idx, (axis_dir, color) in enumerate(zip(axis_dirs_world, axis_colors)):
                start_point = gate_origin.cpu().tolist()
                end_point = (gate_origin + axis_dir).cpu().tolist()
                
                # Draw line from origin to end point
                self.draw.draw_lines(
                    [start_point],
                    [end_point],
                    [color],
                    [3.0]  # Line width
                )

    def _design_scene(self):
        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name, drone_model_cfg.controller
        )

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        # Create gates based on track configuration
        scale = torch.ones(3) * self.gate_scale
        gate_positions_list = []
        gate_orientations_list = []
        
        if self.track_type == "config":
            # Config-based track: gates defined with positions and yaw angles
            # Sort gate keys to ensure correct order
            gate_keys = sorted(self.track_config.keys(), key=lambda x: int(x))
            
            for i, gate_key in enumerate(gate_keys):
                gate_cfg = self.track_config[gate_key]
                pos = gate_cfg.get("pos", (0.0, 0.0, 1.0))
                yaw = gate_cfg.get("yaw", 0.0)
                
                # Convert to torch tensors
                if isinstance(pos, (list, tuple)):
                    gate_pos = torch.tensor(pos, device=self.device, dtype=torch.float32)
                else:
                    gate_pos = torch.tensor([pos[0], pos[1], pos[2]], device=self.device, dtype=torch.float32)
                
                if isinstance(yaw, torch.Tensor):
                    gate_yaw = yaw.item() if yaw.numel() == 1 else yaw
                else:
                    gate_yaw = float(yaw)
                
                # Create quaternion from yaw (rotation around z-axis)
                gate_orientation = euler_to_quaternion(
                    torch.tensor([0., 0., gate_yaw], device=self.device)
                )
                
                gate_positions_list.append(gate_pos)
                gate_orientations_list.append(gate_orientation)
                
                # Spawn gate using configured gate asset
                gate_prim = prim_utils.create_prim(
                    f"/World/envs/env_0/Gate_{i}",
                    usd_path=self.gate_asset_path,
                    translation=(gate_pos[0].item(), gate_pos[1].item(), gate_pos[2].item()),
                    orientation=(gate_orientation[0].item(), gate_orientation[1].item(), 
                                 gate_orientation[2].item(), gate_orientation[3].item()),
                    scale=scale
                )
                # Make gate static: disable gravity and make kinematic
                gate_prim_path = f"/World/envs/env_0/Gate_{i}"
                kit_utils.set_nested_rigid_body_properties(
                    gate_prim_path,
                    disable_gravity=True,
                    linear_damping=1000.0,  # Very high damping to prevent movement
                    angular_damping=1000.0,
                )
                # Set kinematic on all nested rigid bodies to prevent movement from collisions
                gate_prim_obj = prim_utils.get_prim_at_path(gate_prim_path)
                all_prims = [gate_prim_obj]
                while len(all_prims) > 0:
                    child_prim = all_prims.pop(0)
                    if child_prim.HasAttribute("physics:kinematicEnabled"):
                        child_prim.GetAttribute("physics:kinematicEnabled").Set(True)
                    all_prims += child_prim.GetChildren()
        else:
            # Circular track (backward compatibility)
            track_radius_tensor = torch.tensor(self.track_radius, device=self.device, dtype=torch.float32)
            gate_height_tensor = torch.tensor(self.gate_height, device=self.device, dtype=torch.float32)
            num_gates_tensor = torch.tensor(self.num_gates, device=self.device, dtype=torch.float32)
            
            for i in range(self.num_gates):
                # Calculate gate position along circular track
                i_tensor = torch.tensor(i, device=self.device, dtype=torch.float32)
                angle = 2 * torch.pi * i_tensor / num_gates_tensor
                gate_x = track_radius_tensor * torch.cos(angle)
                gate_y = track_radius_tensor * torch.sin(angle)
                gate_z = gate_height_tensor
                
                # Gate orientation: perpendicular to radial direction (tangent to circle)
                # For a circular track, gates should be perpendicular so the drone flies through them in a circle
                # The gate's local x-axis should be tangent to the circle (perpendicular to radial direction)
                # Radial direction: (cos(angle), sin(angle), 0)
                # Tangent direction (perpendicular, rotated 90 degrees): (-sin(angle), cos(angle), 0)
                # So yaw should be angle + pi/2
                gate_yaw = angle + torch.pi / 2
                gate_orientation = euler_to_quaternion(
                    torch.tensor([0., 0., gate_yaw.item()], device=self.device)
                )
                
                gate_positions_list.append(torch.tensor([gate_x.item(), gate_y.item(), gate_z.item()], device=self.device))
                gate_orientations_list.append(gate_orientation)
                
                # Spawn gate using configured gate asset
                gate_prim = prim_utils.create_prim(
                    f"/World/envs/env_0/Gate_{i}",
                    usd_path=self.gate_asset_path,
                    translation=(gate_x.item(), gate_y.item(), gate_z.item()),
                    orientation=(gate_orientation[0].item(), gate_orientation[1].item(), 
                                 gate_orientation[2].item(), gate_orientation[3].item()),
                    scale=scale
                )
                # Make gate static: disable gravity and make kinematic
                gate_prim_path = f"/World/envs/env_0/Gate_{i}"
                kit_utils.set_rigid_body_properties(
                    gate_prim_path,
                    disable_gravity=True,
                    linear_damping=1000.0,  # Very high damping to prevent movement
                    angular_damping=1000.0,
                )
                rigid_api = UsdPhysics.RigidBodyAPI.Apply(gate_prim)
                rigid_api.CreateKinematicEnabledAttr().Set(True)

        # Gate positions and orientations will be retrieved from views at runtime
        # No need to store them manually since gates are static

        # Spawn drone at start position (behind first gate, in gate's local frame)
        # The gate's local x-axis points in the tangent direction (for circular) or forward (for config)
        # We want to spawn the drone behind the gate, so we offset backward along the gate's x-axis
        first_gate_pos = gate_positions_list[0]
        first_gate_rot = gate_orientations_list[0]
        # Offset backward in gate's local frame (negative x direction)
        offset_local = torch.tensor([-1.5, 0.0, 0.0], device=self.device)
        # Rotate offset to world frame - quat_rotate expects batched inputs, so add batch dimension
        offset_world = quat_rotate(first_gate_rot.unsqueeze(0), offset_local.unsqueeze(0)).squeeze(0)
        start_pos = first_gate_pos + offset_world
        # Store first gate position as landing target after completing all laps
        self.first_gate_pos = first_gate_pos
        self.first_gate_rot = first_gate_rot
        self.drone.spawn(translations=[(start_pos[0].item(), start_pos[1].item(), start_pos[2].item())])
        self.lap = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)


        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        # Observation: drone_state[..., 3:] (excludes position) + next_gate_rpos_local (3) + next_gate_rpos_world (3) + gate_progress (1)
        # drone_state[..., 3:] has dimension (drone_state_dim - 3)
        drone_state_dim_no_pos = drone_state_dim - 3
        observation_dim = drone_state_dim_no_pos + 3 + 3 + 1
        if self.time_encoding:
            self.time_encoding_dim = 4
            observation_dim += self.time_encoding_dim
        
        self.observation_spec = Composite({
            "agents": {
                "observation": Unbounded((1, observation_dim), device=self.device),
                "intrinsics": self.drone.intrinsics_spec.unsqueeze(0).to(self.device)
            }
        }).expand(self.num_envs).to(self.device)
        self.action_spec = Composite({
            "agents": {
                "action": self.drone.action_spec.unsqueeze(0),
            }
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = Composite({
            "agents": {
                "reward": Unbounded((1, 1))
            }
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics")
        )
        stats_spec = Composite({
            "return": Unbounded(1),
            "episode_len": Unbounded(1),
            "gates_passed": Unbounded(1),
            "drone_uprightness": Unbounded(1),
            "collision": Unbounded(1),
            "success": BinaryDiscreteTensorSpec(1, dtype=bool),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)
        
        # Reset gate progress
        self.gate_indices[env_ids] = 0
        self.gate_passed[env_ids] = False
        self.lap[env_ids] = 0
        # Reset gate velocities to prevent drift (gates are static, so we just zero velocities)
        # Set velocities to zero for all gates in reset environments
        num_gates_to_reset = len(env_ids) * self.num_gates
        gate_velocities = torch.zeros(num_gates_to_reset, 6, device=self.device)
        # Reshape to match gate view shape: (num_envs, num_gates, 6)
        gate_velocities = gate_velocities.reshape(len(env_ids), self.num_gates, 6)
        self.gates.set_velocities(gate_velocities, env_indices=env_ids)

        # Reset drone position and orientation
        drone_rpy = self.init_rpy_dist.sample((*env_ids.shape, 1))
        drone_rot = euler_to_quaternion(drone_rpy)
        try:
        
            # Position drone near the first gate
            # Get gate positions from views - get all gates first, then select the ones we need
            # This avoids the unflatten issue when using env_indices
            gate_world_pos, gate_world_rot = self.gates.get_world_poses()  # (num_envs, num_gates, 3), (num_envs, num_gates, 4)
            # Select only the environments we're resetting
            # gate_world_pos = gate_world_pos[env_ids]  # (len(env_ids), num_gates, 3)
            # gate_world_rot = gate_world_rot[env_ids]  # (len(env_ids), num_gates, 4)
            gate_env_pos, gate_env_rot = self.get_env_poses((gate_world_pos, gate_world_rot))  # (N, num_gates, 3), (N, num_gates, 4)
            gate_env_pos = gate_env_pos[env_ids]  # (len(env_ids), num_gates, 3)
            gate_env_rot = gate_env_rot[env_ids]  # (len(env_ids), num_gates, 4)
            first_gate_pos = gate_env_pos[:, 0]  # (N, 3)
            first_gate_rot = gate_env_rot[:, 0]  # (N, 4)
            
            # Calculate offset in gate's local frame (behind the gate)
            # Gate's local x-axis points in the forward direction (tangent for circular, or as specified for config)
            # Offset backward along the gate's x-axis
            
            # Rotate offset to world frame using gate's orientation
            # Expand offset_local to match batch size
            offset_local_expanded = self.offset_local.unsqueeze(0).expand(len(env_ids), -1)  # (N, 3)
            offset_world = quat_rotate(first_gate_rot, offset_local_expanded)  # (len(env_ids), 3)
            drone_start_pos = first_gate_pos + offset_world  # (len(env_ids), 3)
            
            # Match fly_through.py pattern exactly: positions should be (len(env_ids), 1, 3)
            # The drone view shape is [num_envs, 1], so positions need the agent dimension
            drone_start_pos_with_agent = drone_start_pos.unsqueeze(1)  # (len(env_ids), 1, 3)
            env_positions_with_agent = self.envs_positions[env_ids].unsqueeze(1)  # (len(env_ids), 1, 3)
            self.drone.set_world_poses(
                drone_start_pos_with_agent + env_positions_with_agent,
                drone_rot, env_ids
            )
        except Exception as e:
            import traceback
            print("=" * 80)
            print(f"ERROR: Failed in _reset_idx (num_envs={self.num_envs}, num_gates={self.num_gates})")
            print("=" * 80)
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {str(e)}")
            print("\nFull traceback:")
            traceback.print_exc()
            print("=" * 80)
            import IPython; IPython.embed(); exit()
        self.drone.set_velocities(
            torch.zeros(len(env_ids), 1, 6, device=self.device), env_ids
        )

        self.drone.set_joint_positions(torch.zeros(len(env_ids), 1, 4, device=self.device), env_ids)
        self.drone.set_joint_velocities(torch.zeros(len(env_ids), 1, 4, device=self.device), env_ids)

        self.stats.exclude("success")[env_ids] = 0.
        self.stats["success"][env_ids] = False

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        import traceback
        import sys
        
        try:
            self.drone_state = self.drone.get_state()
            drone_pos = self.drone_state[..., :3]  # (N, 1, 3) - keep agent dimension
            drone_rot = self.drone_state[..., 3:7]  # (N, 1, 4) - keep agent dimension
            
            # Get gate positions from views (similar to fly_through.py)
            # gates.get_world_poses() returns (pos, rot) with shape (num_envs, num_gates, ...)
            gate_world_pos, gate_world_rot = self.gates.get_world_poses()  # (N, num_gates, 3), (N, num_gates, 4)
        except Exception as e:
            print("=" * 80)
            print(f"ERROR: Failed in _compute_state_and_obs (num_envs={self.num_envs}, num_gates={self.num_gates})")
            print("=" * 80)
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {str(e)}")
            print("\nFull traceback:")
            traceback.print_exc()
            print("=" * 80)
            sys.stderr.write("=" * 80 + "\n")
            sys.stderr.write(f"ERROR: Failed in _compute_state_and_obs\n")
            traceback.print_exc(file=sys.stderr)
            sys.stderr.write("=" * 80 + "\n")
            raise
        gate_env_pos, gate_env_rot = self.get_env_poses((gate_world_pos, gate_world_rot))  # (N, num_gates, 3), (N, num_gates, 4)
        
        # Check if all laps are completed - if so, target first gate for landing
        all_laps_completed = self.lap >= self.num_laps  # (N,)
        
        # Get current gate positions for each environment
        current_gate_indices = self.gate_indices  # (N,)
        batch_indices = torch.arange(self.num_envs, device=self.device)
        
        # If all laps completed, target first gate (index 0) for landing
        # Otherwise, target the current next gate
        target_gate_indices = torch.where(all_laps_completed, 
                                         torch.zeros_like(current_gate_indices),
                                         current_gate_indices)  # (N,)
        
        next_gate_pos = gate_env_pos[batch_indices, target_gate_indices]  # (N, 3)
        next_gate_rot = gate_env_rot[batch_indices, target_gate_indices]  # (N, 4)
        
        # Expand gate positions to match agent dimension for broadcasting
        next_gate_pos = next_gate_pos.unsqueeze(1)  # (N, 1, 3)
        next_gate_rot = next_gate_rot.unsqueeze(1)  # (N, 1, 4)
        
        # Relative position in world frame
        next_gate_rpos_world = next_gate_pos - drone_pos  # (N, 1, 3)
        
        # Relative position in drone's local frame
        # quat_rotate_inverse expects (..., 4) and (..., 3), so we need to handle agent dimension
        drone_rot_flat = drone_rot.squeeze(1)  # (N, 4) for quat_rotate_inverse
        next_gate_rpos_world_flat = next_gate_rpos_world.squeeze(1)  # (N, 3) for quat_rotate_inverse
        next_gate_rpos_local_flat = quat_rotate_inverse(drone_rot_flat, next_gate_rpos_world_flat)  # (N, 3)
        next_gate_rpos_local = next_gate_rpos_local_flat.unsqueeze(1)  # (N, 1, 3)
        
        # Gate progress: combine lap progress and gate progress within current lap
        # Progress = (lap * num_gates + gate_index) / (num_laps * num_gates)
        lap_progress = self.lap.float() * self.num_gates  # (N,)
        gate_progress = (lap_progress + self.gate_indices.float()) / (self.num_laps * self.num_gates)  # (N,)
        
        # Build observation: match the pattern from fly_through.py
        # All components need to have the agent dimension (middle dimension) to match spec (N, 1, obs_dim)
        obs = [
            self.drone_state[..., 3:],  # (N, 1, state_dim-3) - already has agent dimension
            next_gate_rpos_local,  # (N, 1, 3) - already has agent dimension
            next_gate_rpos_world,  # (N, 1, 3) - already has agent dimension
            gate_progress.unsqueeze(1).unsqueeze(-1),  # (N, 1, 1) - add agent and feature dimensions
        ]
        
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length)  # (N,)
            obs.append(t.unsqueeze(1).unsqueeze(1).expand(-1, 1, self.time_encoding_dim))  # (N, 1, time_encoding_dim)
        
        # Concatenate along last dimension: (N, 1, obs_dim)
        obs = torch.cat(obs, dim=-1)  # (N, 1, obs_dim)

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "intrinsics": self.drone.intrinsics,
                },
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        import traceback
        import sys
        
        # Check if all laps are completed - if so, target first gate for landing
        all_laps_completed = self.lap >= self.num_laps  # (N,)
        
        try:
            drone_pos = self.drone_state[..., :3]  # (N, 1, 3) - keep agent dimension
            
            # Get gate positions from views
            gate_world_pos, gate_world_rot = self.gates.get_world_poses()  # (N, num_gates, 3), (N, num_gates, 4)
            gate_env_pos, gate_env_rot = self.get_env_poses((gate_world_pos, gate_world_rot))  # (N, num_gates, 3), (N, num_gates, 4)
            
            # Debug visualization: draw coordinate axes at gate origins (in world coordinates)
            if self.debug_gate_origins:
                self._draw_gate_origins(gate_world_pos, gate_world_rot, env_idx=0)
            
            # Get current gate positions
            batch_indices = torch.arange(self.num_envs, device=self.device)
            
            # For reward computation, use current gate during racing, first gate after all laps completed
            reward_gate_indices = torch.where(all_laps_completed,
                                              torch.zeros_like(self.gate_indices),
                                              self.gate_indices)  # (N,)
            
            current_gate_pos = gate_env_pos[batch_indices, reward_gate_indices]  # (N, 3)
        except Exception as e:
            print("=" * 80)
            print(f"ERROR: Failed in _compute_reward_and_done (num_envs={self.num_envs}, num_gates={self.num_gates})")
            print("=" * 80)
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {str(e)}")
            print("\nFull traceback:")
            traceback.print_exc()
            print("=" * 80)
            sys.stderr.write("=" * 80 + "\n")
            sys.stderr.write(f"ERROR: Failed in _compute_reward_and_done\n")
            traceback.print_exc(file=sys.stderr)
            sys.stderr.write("=" * 80 + "\n")
            raise
        current_gate_rot = gate_env_rot[batch_indices, reward_gate_indices]  # (N, 4)
        
        # Expand gate positions to match agent dimension for broadcasting
        current_gate_pos = current_gate_pos.unsqueeze(1)  # (N, 1, 3)
        current_gate_rot = current_gate_rot.unsqueeze(1)  # (N, 1, 4)
        
        # Calculate gate center: gate origin is at center bottom, so center is at (0, 0, gate_height/2) in gate's local frame
        # Transform gate center offset to world frame
        gate_center_offset_local = torch.tensor([0.0, 0.0, self.gate_height / 2.0], device=self.device)  # (3,)
        gate_center_offset_local_expanded = gate_center_offset_local.unsqueeze(0).expand(self.num_envs, -1)  # (N, 3)
        gate_center_offset_world = quat_rotate(
            current_gate_rot.squeeze(1),  # (N, 4)
            gate_center_offset_local_expanded  # (N, 3)
        )  # (N, 3)
        current_gate_center = current_gate_pos.squeeze(1) + gate_center_offset_world  # (N, 3)
        current_gate_center = current_gate_center.unsqueeze(1)  # (N, 1, 3)
        
        # Distance to gate center (not origin) - squeeze agent dimension for computation
        distance_to_gate = torch.norm((current_gate_center - drone_pos).squeeze(1), dim=-1)  # (N,)
        
        # Check if gate is passed (drone is close enough and has passed through the gate plane)
        # Gate plane is perpendicular to the gate's forward direction (pointing toward track center)
        # Use gate origin for plane crossing check (plane is at origin)
        gate_to_drone = drone_pos - current_gate_pos  # (N, 1, 3)
        # quat_rotate_inverse expects (..., 4) and (..., 3), so squeeze agent dimension
        gate_to_drone_flat = gate_to_drone.squeeze(1)  # (N, 3)
        current_gate_rot_flat = current_gate_rot.squeeze(1)  # (N, 4)
        gate_to_drone_local = quat_rotate_inverse(current_gate_rot_flat, gate_to_drone_flat)  # (N, 3)
        
        # Check if drone has passed through the gate plane (x > 0 in gate's local frame means passed through)
        # The gate's local x-axis points in the forward direction (tangent for circular, or as specified for config)
        passed_plane = gate_to_drone_local[..., 0] > 0  # (N,)
        close_enough = distance_to_gate < self.pass_threshold  # (N,)
        # Only count as passed if we haven't already counted this gate
        gate_passed_this_step = passed_plane & close_enough & (~self.gate_passed)  # (N,)
        
        # Mark gates as passed to prevent detecting the same gate multiple times
        self.gate_passed[gate_passed_this_step] = True
        
        # Update gate indices for environments that passed a gate
        # When moving to a new gate, reset gate_passed flag so we can detect passing the new gate
        old_gate_indices = self.gate_indices.clone()
        self.lap[gate_passed_this_step] = self.lap[gate_passed_this_step] + (self.gate_indices[gate_passed_this_step] + 1 >= self.num_gates).int()
        self.gate_indices[gate_passed_this_step] = (self.gate_indices[gate_passed_this_step] + 1) % self.num_gates
        # Reset gate_passed flag for environments where gate index changed (moved to new gate)
        gate_index_changed = (self.gate_indices != old_gate_indices)  # (N,)
        self.gate_passed[gate_index_changed] = False
        
        # Progress reward: encourage moving toward the gate
        # Use track_radius if available, otherwise use a default value
        track_radius = getattr(self, 'track_radius', 5.0)
        reward_progress = torch.exp(-self.reward_progress_scale * distance_to_gate / track_radius)
        
        # Gate passage reward: only give during racing phase, not during landing phase
        # After all laps are completed, focus on landing reward instead
        reward_gate_passage = gate_passed_this_step.float() * self.reward_gate_passage * (~all_laps_completed).float()
        
        # Track progress reward: combine lap and gate progress
        # Progress = (lap * num_gates + gate_index) / (num_laps * num_gates)
        lap_progress = self.lap.float() * self.num_gates  # (N,)
        track_progress = (lap_progress + self.gate_indices.float()) / (self.num_laps * self.num_gates)  # (N,)
        reward_track_progress = track_progress * 2.0
        
        # Uprightness reward
        drone_up = self.drone_state[..., 16:19]  # (N, 1, 3) - keep agent dimension
        reward_up = 0.5 * torch.square((drone_up.squeeze(1)[..., 2] + 1) / 2)  # (N,)
        
        # Effort reward
        reward_effort = self.reward_effort_weight * torch.exp(-self.effort.squeeze(1))  # (N,)
        
        # Spin reward
        spin = torch.square(self.drone.vel[..., -1].squeeze(1))  # (N,)
        reward_spin = 0.5 / (1.0 + torch.square(spin))
        
        # Total reward - all rewards are (N,) shape
        reward = (
            reward_progress
            + reward_gate_passage
            + reward_track_progress
            + (reward_up + reward_spin)
            + reward_effort
        )  # (N,)
        
        # Check for collisions if enabled
        collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.reset_on_collision and self.gates is not None:
            contact_forces = self.gates.get_net_contact_forces()
            if contact_forces is not None:
                # contact_forces shape: (num_envs, num_gates, ...)
                # Reduce all dimensions except the first (env dimension) to get (N,)
                while contact_forces.dim() > 1:
                    contact_forces = contact_forces.any(-1)
                collision = contact_forces  # (N,)
        
        # Termination conditions - use self.drone.pos like fly_through.py to get (N, 1, 3) shape
        # In fly_through.py, self.drone.pos is (N, 1, 3), so drone_pos[..., 2] is (N, 1)
        # Use self.drone.pos instead of drone_pos to match fly_through.py exactly
        drone_pos_for_termination = self.drone.pos  # (N, 1, 3) - match fly_through.py
        misbehave = (
            (drone_pos_for_termination[..., 2] < 0.2)  # Too close to ground - (N, 1)
            | (drone_pos_for_termination[..., 2] > 10.0)  # Too high - (N, 1)
        )
        # Add distance check if using circular track
        if self.track_type == "circular":
            # Compute norm on (N, 1, 2) -> (N, 1)
            misbehave |= (torch.norm(drone_pos_for_termination[..., :2], dim=-1) > self.track_radius * 2.0)  # (N, 1)
        else:
            # For config-based tracks, check if drone is too far from any gate
            # Get all gate positions to compute max distance
            gate_world_pos, gate_world_rot = self.gates.get_world_poses()  # (N, num_gates, 3), (N, num_gates, 4)
            gate_env_pos, _ = self.get_env_poses((gate_world_pos, gate_world_rot))  # (N, num_gates, 3), (N, num_gates, 4)
            max_gate_distance = torch.norm(gate_env_pos, dim=-1).max()  # Max distance from origin to any gate
            # Compute norm on (N, 1, 2) -> (N, 1)
            misbehave |= (torch.norm(drone_pos_for_termination[..., :2], dim=-1) > max_gate_distance * 2.0)  # (N, 1)
        # Keep agent dimension like fly_through.py: self.drone_state is (N, 1, state_dim), .any(-1) gives (N, 1)
        hasnan = torch.isnan(self.drone_state).any(-1)  # (N, 1) - match fly_through.py
        
        terminated = misbehave | hasnan  # (N, 1) | (N, 1) -> (N, 1) - match fly_through.py pattern
        if self.reset_on_collision:
            # collision is (N,), need to unsqueeze to (N, 1) to match terminated shape
            terminated |= collision.unsqueeze(-1)  # (N, 1)
        
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)  # (N, 1) - match fly_through.py pattern
        
        done = terminated | truncated
        
        # Success: completed all required laps and landed near first gate
        all_laps_completed = self.lap >= self.num_laps  # (N,)
        
        # Get first gate position in env frame for landing check
        first_gate_pos_env = gate_env_pos[batch_indices, torch.zeros_like(batch_indices)]  # (N, 3) - first gate (index 0)
        first_gate_pos_env = first_gate_pos_env.unsqueeze(1)  # (N, 1, 3)
        
        # Check if landed near first gate (within 0.5m horizontally and below 0.5m height)
        distance_to_first_gate = torch.norm((drone_pos_for_termination - first_gate_pos_env)[..., :2], dim=-1).squeeze(-1)  # (N,) - horizontal distance
        height_above_first_gate = (drone_pos_for_termination[..., 2] - first_gate_pos_env[..., 2]).squeeze(-1)  # (N,) - height difference
        landed_near_first_gate = (distance_to_first_gate < 0.5) & (height_above_first_gate < 0.5) & (height_above_first_gate >= -0.2)  # (N,)
        
        completed_task = all_laps_completed & landed_near_first_gate  # (N,)
        self.stats["success"].bitwise_or_(completed_task.unsqueeze(-1))

        # Reward for approaching first gate after completing all laps (landing phase)
        # Only apply when all laps are completed
        distance_to_first_gate_3d = torch.norm((drone_pos_for_termination - first_gate_pos_env).squeeze(1), dim=-1)  # (N,)
        landing_reward = -distance_to_first_gate_3d * self.reward_progress_scale * 0.5  # Reduced scale for landing
        reward += landing_reward * all_laps_completed.float()

        # Additional landing reward for successfully landing near first gate
        reward += landed_near_first_gate.float() * self.reward_final_position * all_laps_completed.float()
        
        # Add reward to stats - reward is (N,), stats["return"] is (N, 1), so we need to unsqueeze
        self.stats["return"].add_(reward.unsqueeze(-1))
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        if self.reset_on_collision:
            self.stats["collision"].add_(collision.float().unsqueeze(-1))

        # Update stats - squeeze agent dimension for stats updates
        self.stats["gates_passed"][:] = self.gate_indices.float().unsqueeze(1)
        self.stats["drone_uprightness"].mul_(self.alpha).add_((1 - self.alpha) * drone_up.squeeze(1)[..., 2].unsqueeze(-1))

        # Match fly_through.py pattern: terminated is (N,), truncated is (N, 1), done broadcasts to (N, 1)
        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1).unsqueeze(-1),  # (N, 1, 1) to match reward spec
                },
                "done": done,  # (N,) | (N, 1) -> (N, 1) via broadcasting
                "terminated": terminated,  # (N,) - match fly_through.py
                "truncated": truncated,  # (N, 1)
            },
            self.batch_size,
        )
