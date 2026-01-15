"""
Keyboard and Joystick-controlled drone example.

This script allows you to control a single drone in Isaac Lab using keyboard or joystick input.
The drone is controlled using position-based control (similar to PX4 Position Mode).

Keyboard Controls (similar to PX4 Position Mode):
    W/S: Forward/backward velocity (pitch stick)
    A/D: Left/right velocity (roll stick)
    Q/E: Yaw rate (yaw stick)
    Space: Ascend (throttle up)
    Shift: Descend (throttle down)
    R: Reset drone to initial position
    ESC: Exit

Joystick Controls (Logitech gamepad):
    Left Stick X: Roll (left/right velocity)
    Left Stick Y: Pitch (forward/backward velocity) - inverted
    Right Stick X: Yaw rate
    Right Trigger: Ascend (throttle up)
    Left Trigger: Descend (throttle down)
    A Button: Reset drone
    B Button: Exit

When sticks are centered/keys are released, the drone will hold its current position (position hold mode).
"""

import torch
import threading
from collections import defaultdict

import hydra
from omegaconf import OmegaConf, DictConfig
from omni_drones import init_simulation_app

try:
    from pynput import keyboard
    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False
    print("Warning: pynput not installed. Install it with: pip install pynput")
    print("Keyboard control will not work.")

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    print("Warning: pygame not installed. Install it with: pip install pygame")
    print("Joystick control will not work.")


class KeyboardController:
    """Thread-safe keyboard controller for position-based drone control."""
    
    def __init__(self, max_vel_horizontal=2.0, max_vel_vertical=1.5, max_yaw_rate=1.0):
        self.pressed_keys = defaultdict(bool)
        self.lock = threading.Lock()
        self.max_vel_horizontal = max_vel_horizontal  # Maximum horizontal velocity in m/s
        self.max_vel_vertical = max_vel_vertical  # Maximum vertical velocity in m/s
        self.max_yaw_rate = max_yaw_rate  # Maximum yaw rate in rad/s
        
        # Key mappings
        self.key_mappings = {
            'w': 'forward',
            's': 'backward',
            'a': 'left',
            'd': 'right',
            'q': 'yaw_left',
            'e': 'yaw_right',
            'space': 'ascend',
            'shift': 'descend',
            'r': 'reset',
            'esc': 'exit',
        }
        
        self.listener = None
        if HAS_PYNPUT:
            self._start_listener()
    
    def _start_listener(self):
        """Start the keyboard listener in a separate thread."""
        def on_press(key):
            try:
                key_name = key.char if hasattr(key, 'char') and key.char else None
                if key_name:
                    with self.lock:
                        self.pressed_keys[key_name] = True
                elif key == keyboard.Key.space:
                    with self.lock:
                        self.pressed_keys['space'] = True
                elif key == keyboard.Key.shift:
                    with self.lock:
                        self.pressed_keys['shift'] = True
                elif key == keyboard.Key.esc:
                    with self.lock:
                        self.pressed_keys['esc'] = True
            except AttributeError:
                pass
        
        def on_release(key):
            try:
                key_name = key.char if hasattr(key, 'char') and key.char else None
                if key_name:
                    with self.lock:
                        self.pressed_keys[key_name] = False
                elif key == keyboard.Key.space:
                    with self.lock:
                        self.pressed_keys['space'] = False
                elif key == keyboard.Key.shift:
                    with self.lock:
                        self.pressed_keys['shift'] = False
            except AttributeError:
                pass
        
        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()
    
    def get_commands(self, dt, current_pos, current_yaw):
        """
        Get velocity and yaw rate commands based on current keyboard state.
        Similar to PX4 Position Mode: centered sticks hold position.
        
        Args:
            dt: Time step
            current_pos: Current position [x, y, z]
            current_yaw: Current yaw angle in radians
        
        Returns:
            target_vel: Target velocity in world frame [vx, vy, vz]
            target_yaw_rate: Target yaw rate in rad/s
            reset: Whether to reset
            exit_flag: Whether to exit
        """
        with self.lock:
            pressed = dict(self.pressed_keys)
        
        # Initialize velocity commands (zero = position hold)
        target_vel = torch.zeros(3)
        target_yaw_rate = 0.0
        
        # Compute forward/backward and left/right directions based on current yaw
        cos_yaw = torch.cos(torch.tensor(current_yaw))
        sin_yaw = torch.sin(torch.tensor(current_yaw))
        
        # Process key presses - similar to PX4 Position Mode
        # Compute body frame velocity components first
        vel_forward = 0.0
        vel_right = 0.0
        
        # W/S: Forward/backward velocity (in body frame)
        if pressed.get('w', False):
            vel_forward = self.max_vel_horizontal
        elif pressed.get('s', False):
            vel_forward = -self.max_vel_horizontal
        
        # A/D: Left/right velocity (in body frame)
        if pressed.get('a', False):
            vel_right = -self.max_vel_horizontal  # Left is negative
        elif pressed.get('d', False):
            vel_right = self.max_vel_horizontal  # Right is positive
        
        # Convert body frame velocity to world frame
        # Forward direction in world frame: (cos_yaw, sin_yaw, 0)
        # Right direction in world frame: (-sin_yaw, cos_yaw, 0)
        target_vel[0] = vel_forward * cos_yaw - vel_right * sin_yaw
        target_vel[1] = vel_forward * sin_yaw + vel_right * cos_yaw
        
        # Q/E: Yaw rate
        if pressed.get('q', False):
            target_yaw_rate = -self.max_yaw_rate  # Yaw left
        if pressed.get('e', False):
            target_yaw_rate = self.max_yaw_rate  # Yaw right
        
        # Space/Shift: Vertical velocity
        if pressed.get('space', False):
            target_vel[2] = self.max_vel_vertical  # Ascend
        if pressed.get('shift', False):
            target_vel[2] = -self.max_vel_vertical  # Descend
        
        # Check for reset or exit
        reset = pressed.get('r', False)
        exit_flag = pressed.get('esc', False)
        
        return target_vel, target_yaw_rate, reset, exit_flag
    
    def stop(self):
        """Stop the keyboard listener."""
        if self.listener:
            self.listener.stop()


class JoystickController:
    """Joystick controller for position-based drone control using pygame."""
    
    def __init__(self, max_vel_horizontal=2.0, max_vel_vertical=1.5, max_yaw_rate=1.0, 
                 deadzone=0.1, joystick_id=0):
        self.max_vel_horizontal = max_vel_horizontal
        self.max_vel_vertical = max_vel_vertical
        self.max_yaw_rate = max_yaw_rate
        self.deadzone = deadzone  # Deadzone for stick inputs
        self.joystick_id = joystick_id
        
        self.joystick = None
        self.initialized = False
        
        if HAS_PYGAME:
            self._init_joystick()
    
    def _init_joystick(self):
        """Initialize pygame and joystick."""
        try:
            # Initialize pygame with joystick support
            pygame.init()
            
            # Explicitly initialize joystick module
            if not pygame.joystick.get_init():
                pygame.joystick.init()
            
            # Check if joysticks are available
            joystick_count = pygame.joystick.get_count()
            print(f"Pygame detected {joystick_count} joystick(s)")
            
            if joystick_count == 0:
                print("Warning: No joysticks detected by pygame.")
                print("  Make sure the joystick is connected and try:")
                print("  - Unplugging and replugging the joystick")
                print("  - Checking permissions (may need to add user to input group)")
                print("  - Running: sudo usermod -a -G input $USER (then logout/login)")
                return
            
            # List all available joysticks
            for i in range(joystick_count):
                temp_joy = pygame.joystick.Joystick(i)
                temp_joy.init()
                print(f"  Joystick {i}: {temp_joy.get_name()}")
                temp_joy.quit()
            
            if self.joystick_id >= joystick_count:
                print(f"Warning: Joystick {self.joystick_id} not available. Using joystick 0.")
                self.joystick_id = 0
            
            self.joystick = pygame.joystick.Joystick(self.joystick_id)
            self.joystick.init()
            
            print(f"\n✓ Joystick initialized: {self.joystick.get_name()}")
            print(f"  Axes: {self.joystick.get_numaxes()}")
            print(f"  Buttons: {self.joystick.get_numbuttons()}")
            print(f"  Hats: {self.joystick.get_numhats()}")
            print(f"  Balls: {self.joystick.get_numballs()}")
            self.initialized = True
        except Exception as e:
            import traceback
            print(f"Error initializing joystick: {e}")
            print(traceback.format_exc())
            self.initialized = False
    
    def get_commands(self, dt, current_pos, current_yaw):
        """
        Get velocity and yaw rate commands based on current joystick state.
        Similar to PX4 Position Mode: centered sticks hold position.
        
        Args:
            dt: Time step
            current_pos: Current position [x, y, z]
            current_yaw: Current yaw angle in radians
        
        Returns:
            target_vel: Target velocity in world frame [vx, vy, vz]
            target_yaw_rate: Target yaw rate in rad/s
            reset: Whether to reset
            exit_flag: Whether to exit
        """
        if not self.initialized or self.joystick is None:
            return torch.zeros(3), 0.0, False, False
        
        # Process pygame events (required for joystick input)
        # This is critical - pygame needs to process events to update joystick state
        for event in pygame.event.get():
            pass  # Process all events but don't need to handle them
        
        # Initialize velocity commands (zero = position hold)
        target_vel = torch.zeros(3)
        target_yaw_rate = 0.0
        
        # Get stick values (normalized to [-1, 1])
        # For Logitech F310 in XInput mode:
        # - Left stick: axes 0 (X/roll), 1 (Y/pitch)
        # - Right stick: axes 2 (X/yaw), 3 (Y/unused)
        # - Triggers: axes 4 (left), 5 (right) - but may be combined as axis 2
        try:
            num_axes = self.joystick.get_numaxes()
            
            # Left stick
            if num_axes > 0:
                left_stick_x = -self.joystick.get_axis(0)  # Roll (left/right)
            else:
                left_stick_x = 0.0
            
            if num_axes > 1:
                left_stick_y = -self.joystick.get_axis(1)  # Pitch (forward/back)
            else:
                left_stick_y = 0.0
            
            # Right stick for yaw (XInput mode typically uses axes 2 and 3)
            # But some controllers combine triggers, so check axis count
            if num_axes > 3:
                # XInput mode: right stick is axes 2 and 3
                right_stick_x = -self.joystick.get_axis(3)  # Yaw (inverted)
            elif num_axes > 2:
                # Might be axis 2 for yaw
                right_stick_x = -self.joystick.get_axis(2)  # Yaw (inverted)
            else:
                right_stick_x = 0.0
            
            # Apply deadzone - ensure values are exactly 0.0 when within deadzone
            if abs(left_stick_x) < self.deadzone:
                left_stick_x = 0.0
            if abs(left_stick_y) < self.deadzone:
                left_stick_y = 0.0
            # For yaw, use a slightly larger deadzone to prevent drift
            # yaw_deadzone = max(self.deadzone, 0.15)  # At least 15% deadzone for yaw
            yaw_deadzone = 0.00390625
            if abs(right_stick_x) < yaw_deadzone:
                right_stick_x = 0.0
            # Additional safety check: clamp very small values to zero
            if abs(right_stick_x) < 0.05:
                right_stick_x = 0.0
            
            # Triggers for vertical velocity
            # For Logitech F310 XInput mode:
            # - Triggers might be on axes 4 and 5, or combined
            # - Right trigger (ascend) and left trigger (descend)
            vertical_input = 0.0
            num_axes = self.joystick.get_numaxes()
            
            if num_axes > 5:

                # Separate trigger axes (XInput mode)
                # Left trigger: axis 2, Right trigger: axis 5
                # Triggers: -1.0 at rest, +1.0 when fully pressed
                left_trigger_raw = self.joystick.get_axis(2)
                right_trigger_raw = self.joystick.get_axis(5)
                
                # Normalize: convert from -1.0 (rest) to +1.0 (pressed) range to 0.0 to 1.0 range
                # Formula: (raw + 1.0) / 2.0
                # -1.0 (rest) → 0.0 (no input)
                # +1.0 (pressed) → 1.0 (max input)
                right_trigger_normalized = (right_trigger_raw + 1.0) / 2.0
                left_trigger_normalized = (left_trigger_raw + 1.0) / 2.0
                
                # Apply deadzone: if trigger is close to rest position (-1.0), treat as 0
                # Check if raw value is within deadzone of -1.0 (rest position)
                # Use a deadzone of 0.1, so if raw < -0.9, treat as at rest
                trigger_deadzone = 0.0
                raw_rest_threshold = -1.0 + trigger_deadzone  # -0.9
                
                if right_trigger_raw < raw_rest_threshold:
                    # Trigger is at or near rest position, set to 0
                    right_trigger = 0.0
                else:
                    # Trigger is pressed, use normalized value (clamped to [0, 1])
                    right_trigger = max(0.0, min(1.0, right_trigger_normalized))
                
                if left_trigger_raw < raw_rest_threshold:
                    # Trigger is at or near rest position, set to 0
                    left_trigger = 0.0
                else:
                    # Trigger is pressed, use normalized value (clamped to [0, 1])
                    left_trigger = max(0.0, min(1.0, left_trigger_normalized))
                
                # Right trigger positive (ascend), left trigger negative (descend)
                vertical_input = right_trigger - left_trigger
                
                # Final safety check: if vertical input is very small, set to exactly 0
                if abs(vertical_input) < 0.05:
                    vertical_input = 0.0
            elif num_axes > 4:
                # Might have one trigger axis
                trigger = self.joystick.get_axis(4)
                # Apply deadzone
                if abs(trigger) < self.deadzone:
                    trigger = 0.0
                vertical_input = trigger
            else:
                # Fallback: use buttons or hat for vertical control
                # Button 0 (A) for up, Button 1 (B) for down
                num_buttons = self.joystick.get_numbuttons()
                if num_buttons > 0:
                    if self.joystick.get_button(0):  # A button
                        vertical_input = 1.0
                    elif num_buttons > 1 and self.joystick.get_button(1):  # B button
                        vertical_input = -1.0
                
                # Also check hat (D-pad) if available
                if self.joystick.get_numhats() > 0:
                    hat = self.joystick.get_hat(0)
                    if hat[1] > 0:  # Up on hat
                        vertical_input = 1.0
                    elif hat[1] < 0:  # Down on hat
                        vertical_input = -1.0
            
            # Compute body frame velocity components
            vel_forward = left_stick_y * self.max_vel_horizontal
            vel_right = left_stick_x * self.max_vel_horizontal
            
            # Convert body frame velocity to world frame
            cos_yaw = torch.cos(torch.tensor(current_yaw))
            sin_yaw = torch.sin(torch.tensor(current_yaw))
            target_vel[0] = vel_forward * cos_yaw - vel_right * sin_yaw
            target_vel[1] = vel_forward * sin_yaw + vel_right * cos_yaw
            
            # Yaw rate from right stick - ensure it's exactly 0.0 when stick is centered
            # Only compute yaw rate if stick is outside deadzone
            if abs(right_stick_x) < 0.05:  # Use stricter threshold for yaw
                target_yaw_rate = 0.0
            else:
                target_yaw_rate = right_stick_x * self.max_yaw_rate
                # Additional clamp for very small values
                if abs(target_yaw_rate) < 0.01:
                    target_yaw_rate = 0.0
            
            # Vertical velocity from triggers
            target_vel[2] = vertical_input * self.max_vel_vertical
            
            # Check buttons
            reset = False
            exit_flag = False
            if self.joystick.get_numbuttons() > 0:
                # Button 0 (A) - Reset (if not used for vertical)
                if self.joystick.get_numaxes() <= 4 and self.joystick.get_button(0):
                    reset = True
                # Button 1 (B) - Exit (if not used for vertical)
                elif self.joystick.get_numaxes() <= 4 and self.joystick.get_button(1):
                    exit_flag = True
                # Button 2 (X) - Reset
                elif self.joystick.get_numbuttons() > 2 and self.joystick.get_button(2):
                    reset = True
                # Button 3 (Y) - Exit
                elif self.joystick.get_numbuttons() > 3 and self.joystick.get_button(3):
                    exit_flag = True
            
        except Exception as e:
            print(f"Error reading joystick: {e}")
            return torch.zeros(3), 0.0, False, False
        
        return target_vel, target_yaw_rate, reset, exit_flag
    
    def stop(self):
        """Stop the joystick."""
        if self.joystick:
            self.joystick.quit()
        if HAS_PYGAME:
            pygame.quit()


@hydra.main(version_base=None, config_path=".", config_name="keyboard_control")
def main(cfg):
    OmegaConf.resolve(cfg)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))
    
    if not HAS_PYNPUT and not (HAS_PYGAME and pygame.joystick.get_count() > 0):
        print("Error: Either pynput (for keyboard) or pygame with joystick is required.")
        print("Install with: pip install pynput  # for keyboard")
        print("          or: pip install pygame  # for joystick")
        simulation_app.close()
        return

    import omni_drones.utils.scene as scene_utils
    from isaacsim.core.api.simulation_context import SimulationContext
    from omni_drones.robots.drone import MultirotorBase
    from omni_drones.utils.torch import euler_to_quaternion, quaternion_to_euler
    
    # Check if Forest environment should be used
    use_forest = cfg.get("use_forest", False)
    forest_env = None
    
    if use_forest:
        # Use Forest environment - it will set up terrain, obstacles, and lidar
        try:
            from omni_drones.envs.single.forest import Forest
            
            # Create a config for Forest environment with single env
            forest_cfg = OmegaConf.create({
                "task": {
                    "name": "Forest",
                    "drone_model": cfg.drone_model,
                    "lidar_range": cfg.get("lidar_range", 4.0),
                    "lidar_vfov": cfg.get("lidar_vfov", [-10., 20.]),
                    "time_encoding": cfg.get("time_encoding", False),
                    "reward_effort_weight": cfg.get("reward_effort_weight", 0.1),
                    "randomization": cfg.get("randomization", {}),
                },
                "env": {
                    "num_envs": 1,  # Single environment for manual control
                    "max_episode_length": cfg.get("steps", 10000),
                    "env_spacing": 1.0,
                },
                "sim": cfg.sim,
                "viewer": cfg.viewer,
            })
            
            # Create Forest environment instance
            forest_env = Forest(forest_cfg, headless=cfg.headless)
            
            # Extract components from Forest environment
            sim = forest_env.sim
            drone = forest_env.drone
            controller = forest_env.controller
            # Ensure controller is on the correct device
            controller = controller.to(cfg.sim.device)
            
            print("\n" + "="*60)
            print("Forest environment enabled!")
            print("  - Terrain with obstacles loaded")
            print("  - Lidar sensor initialized")
            print("="*60 + "\n")
            
        except ImportError as e:
            print(f"Warning: Could not import Forest environment: {e}")
            print("Falling back to basic scene.")
            use_forest = False
    
    if not use_forest:
        # Use basic scene setup
        sim = SimulationContext(
            stage_units_in_meters=1.0,
            physics_dt=cfg.sim.dt,
            rendering_dt=cfg.sim.dt,
            sim_params=cfg.sim,
            backend="torch",
            device=cfg.sim.device,
        )

        # Create a single drone with position controller
        drone_model_cfg = cfg.drone_model
        drone, controller = MultirotorBase.make(
            drone_model_cfg.name, "LeePositionController", cfg.sim.device
        )
        # Ensure controller is on the correct device
        controller = controller.to(cfg.sim.device)

        # Spawn drone at origin, slightly above ground
        translations = torch.zeros(1, 3, device=cfg.sim.device)
        translations[0, 2] = 1.0  # Start at height 1.0m
        drone.spawn(translations=translations)

        scene_utils.design_scene()
    
    # Import OpenCV for real-time camera visualization
    try:
        import cv2
        HAS_OPENCV = True
    except ImportError:
        HAS_OPENCV = False
        print("Warning: OpenCV (cv2) not installed. Install it with: pip install opencv-python")
        print("Real-time camera visualization will not work.")
    from omni_drones.sensors.camera import Camera, PinholeCameraCfg
    import dataclasses

    camera_cfg = PinholeCameraCfg(
        sensor_tick=0,
        resolution=(320, 240),
        data_types=["rgb"],
    )
    # cameras used as sensors
    camera_sensor = Camera(camera_cfg)
    camera_sensor.spawn([
        f"/World/envs/env_0/{drone.name}_0/base_link/Camera"
    ])
    # camera for visualization
    # here we reuse the viewport camera, i.e., "/OmniverseKit_Persp"
    camera_vis = Camera(dataclasses.replace(camera_cfg, resolution=(960, 720)))

    sim.reset()
    camera_sensor.initialize(f"/World/envs/env_0/{drone.name}_*/base_link/Camera")
    camera_vis.initialize("/OmniverseKit_Persp")
    drone.initialize()
    
    # Initialize lidar if using Forest environment
    if use_forest:
        forest_env.lidar._initialize_impl()

    # Initialize controllers for position-based control
    keyboard_ctrl = KeyboardController(
        max_vel_horizontal=cfg.get("max_vel_horizontal", 2.0),
        max_vel_vertical=cfg.get("max_vel_vertical", 1.5),
        max_yaw_rate=cfg.get("max_yaw_rate", 1.0)
    )
    
    # Initialize joystick controller (if available)
    joystick_ctrl = None
    if HAS_PYGAME:
        joystick_ctrl = JoystickController(
            max_vel_horizontal=cfg.get("max_vel_horizontal", 2.0),
            max_vel_vertical=cfg.get("max_vel_vertical", 1.5),
            max_yaw_rate=cfg.get("max_yaw_rate", 1.0),
            deadzone=cfg.get("joystick_deadzone", 0.1),
            joystick_id=cfg.get("joystick_id", 0)
        )
        if not joystick_ctrl.initialized:
            joystick_ctrl = None
            print("Falling back to keyboard control.")
        else:
            print("\n" + "="*60)
            print("Joystick successfully initialized and will be used for control.")
            print("="*60 + "\n")

    # Store initial state for reset
    if use_forest:
        # Get initial pose from Forest environment
        # init_poses is a tuple (positions, orientations) from get_world_poses()
        poses_tuple = forest_env.init_poses
        init_pos = poses_tuple[0][0:1]  # positions tensor, first env
        init_rot = poses_tuple[1][0:1]  # orientations tensor, first env
        init_vels = forest_env.init_vels[0:1]
    else:
        # Use the translations from basic scene setup
        translations = torch.zeros(1, 3, device=cfg.sim.device)
        translations[0, 2] = 1.0  # Start at height 1.0m
        init_pos = translations.clone()
        init_rpy = torch.zeros(1, 3, device=cfg.sim.device)
        init_rot = euler_to_quaternion(init_rpy)
        init_vels = torch.zeros(1, 6, device=cfg.sim.device)

    def reset_drone():
        """Reset drone to initial position."""
        drone._reset_idx(torch.tensor([0], device=cfg.sim.device))
        drone.set_world_poses(init_pos, init_rot)
        drone.set_velocities(init_vels)

    reset_drone()
    drone_state = drone.get_state()[..., :13].squeeze(0)
    
    # Track target yaw for position mode (integrates yaw rate commands)
    # Initialize to current yaw - get rotation from world poses
    pos, rot = drone.get_world_poses(True)
    # Ensure rot has shape [1, 4] for quaternion_to_euler
    # Handle different possible shapes: [4], [1, 4], [1, 1, 4], etc.
    rot_flat = rot.flatten()  # Flatten to 1D
    rot = rot_flat[:4].unsqueeze(0)  # Take first 4 elements and add batch dim -> [1, 4]
    current_rpy = quaternion_to_euler(rot)  # Returns [1, 3] for [roll, pitch, yaw]
    # Verify shape is correct
    if current_rpy.shape != (1, 3):
        raise ValueError(f"quaternion_to_euler returned unexpected shape: {current_rpy.shape}, expected (1, 3)")
    target_yaw_state = torch.tensor([current_rpy[0, 2].item()], device=cfg.sim.device)

    print("\n" + "="*60)
    if joystick_ctrl and joystick_ctrl.initialized:
        print("Joystick Controls (Position Mode - similar to PX4):")
        print("  Left Stick X: Roll (left/right velocity)")
        print("  Left Stick Y: Pitch (forward/backward velocity)")
        print("  Right Stick X: Yaw rate (rotation)")
        print("  Right Trigger: Ascend (vertical velocity up)")
        print("  Left Trigger: Descend (vertical velocity down)")
        print("  X Button: Reset drone")
        print("  Y Button: Exit")
        print("\nNote: Centered sticks will hold current position (position hold)")
    else:
        print("Keyboard Controls (Position Mode - similar to PX4):")
        print("  W/S: Forward/backward velocity")
        print("  A/D: Left/right velocity")
        print("  Q/E: Yaw rate (rotation)")
        print("  Space: Ascend (vertical velocity up)")
        print("  Shift: Descend (vertical velocity down)")
        print("  R: Reset drone")
        print("  ESC: Exit")
        print("\nNote: Releasing keys will hold current position (position hold)")
    print("="*60 + "\n")

    from tqdm import tqdm
    exit_simulation = False
    frames_sensor = []
    frames_vis = []
    # Main simulation loop
    for i in tqdm(range(cfg.get("steps", 10000)), desc="Simulating"):
        if sim.is_stopped() or exit_simulation:
            break
        if not sim.is_playing():
            sim.render()
            continue

        # Get current position and yaw for position-based control
        # Get rotation from world poses
        pos, rot = drone.get_world_poses(True)
        current_pos = pos[0].cpu().numpy()
        # Ensure rot has shape [1, 4] for quaternion_to_euler
        # Handle different possible shapes: [4], [1, 4], [1, 1, 4], etc.
        rot_flat = rot.flatten()  # Flatten to 1D
        rot = rot_flat[:4].unsqueeze(0)  # Take first 4 elements and add batch dim -> [1, 4]
        current_rpy = quaternion_to_euler(rot)  # Returns [1, 3] for [roll, pitch, yaw]
        current_yaw = current_rpy[0, 2].item()
        
        # Get commands from joystick (if available) or keyboard
        if joystick_ctrl and joystick_ctrl.initialized:
            target_vel, target_yaw_rate, reset, exit_flag = joystick_ctrl.get_commands(
                cfg.sim.dt, current_pos, current_yaw
            )
        else:
            target_vel, target_yaw_rate, reset, exit_flag = keyboard_ctrl.get_commands(
                cfg.sim.dt, current_pos, current_yaw
            )
        
        if exit_flag:
            exit_simulation = True
            break
        
        if reset:
            reset_drone()
            drone_state = drone.get_state()[..., :13].squeeze(0)
            # Reset target yaw to initial yaw (0)
            target_yaw_state = torch.tensor([0.0], device=cfg.sim.device)
            continue

        # Integrate yaw rate to get target yaw (similar to PX4 Position Mode)
        # Ensure the increment is a tensor on the correct device
        yaw_increment = torch.tensor(target_yaw_rate * cfg.sim.dt, device=cfg.sim.device)
        target_yaw_state = target_yaw_state + yaw_increment
        
        # Convert to tensors and ensure they're on the correct device
        target_vel_tensor = target_vel.unsqueeze(0).to(cfg.sim.device)
        target_yaw = target_yaw_state.unsqueeze(0).to(cfg.sim.device)

        # Compute control action using LeePositionController
        # When target_vel is zero, controller will hold current position
        # Ensure drone_state is on the correct device
        drone_state_tensor = drone_state.unsqueeze(0).to(cfg.sim.device)
        action = controller.compute(
            drone_state_tensor,
            target_pos=None,  # None means use current position as target (position hold)
            target_vel=target_vel_tensor,  # Velocity command
            target_yaw=target_yaw  # Yaw command
        )

        # Apply action
        drone.apply_action(action)
        
        # Update lidar if using Forest environment
        if use_forest:
            forest_env.lidar.update(sim.get_physics_dt())
        
        sim.step(render=True)

        # Get and display camera images in real-time
        # if i % 2 == 0:  # Update display every 2 frames for performance
        #     sensor_images = camera_sensor.get_images().cpu()
        #     vis_images = camera_vis.get_images().cpu()
            
            # # Store frames for potential video saving
            # frames_sensor.append(sensor_images)
            # frames_vis.append(vis_images)
            
            # # Display camera feed in real-time using OpenCV
            # if HAS_OPENCV:
            #     # Get RGB images from sensor camera
            #     # sensor_images["rgb"] has shape [num_cameras, C, H, W] = [1, C, H, W] for single camera
            #     if "rgb" in sensor_images:
            #         rgb_sensor = sensor_images["rgb"][0]  # Get first (and only) camera, shape: [C, H, W]
            #         # Convert from CHW to HWC format
            #         rgb_sensor = rgb_sensor.permute(1, 2, 0)  # [H, W, C]
            #         # Handle RGBA -> RGB if needed
            #         if rgb_sensor.shape[2] == 4:
            #             rgb_sensor = rgb_sensor[..., :3]
            #         # Convert to numpy and scale to [0, 255] (assuming input is in [0, 1] range)
            #         # If already in [0, 255] range, remove the * 255
            #         rgb_sensor_np = rgb_sensor.numpy()
            #         if rgb_sensor_np.max() <= 1.0:
            #             rgb_sensor_np = (rgb_sensor_np * 255).astype('uint8')
            #         else:
            #             rgb_sensor_np = rgb_sensor_np.astype('uint8')
            #         # Convert RGB to BGR for OpenCV
            #         rgb_sensor_bgr = cv2.cvtColor(rgb_sensor_np, cv2.COLOR_RGB2BGR)
            #         cv2.imshow("Drone Camera Feed", rgb_sensor_bgr)
            #         cv2.waitKey(1)  # Non-blocking wait for key press

        # Update drone state
        drone_state = drone.get_state()[..., :13].squeeze(0)

    # Cleanup
    keyboard_ctrl.stop()
    if joystick_ctrl:
        joystick_ctrl.stop()
    if use_forest:
        # Forest environment cleanup is handled by simulation_app.close()
        pass
    simulation_app.close()


if __name__ == "__main__":
    main()
