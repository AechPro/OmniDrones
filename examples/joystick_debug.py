"""
Simple joystick debug script to capture raw joystick values.

This script helps identify the actual deadzone values needed for your joystick
by displaying all axes, buttons, and hat values in real-time.

Usage:
    python joystick_debug.py

Press Ctrl+C to exit.
"""

import sys
import time

try:
    import pygame
except ImportError:
    print("Error: pygame is required. Install it with: pip install pygame")
    sys.exit(1)


def main():
    # Initialize pygame
    pygame.init()
    pygame.joystick.init()
    
    # Check for joysticks
    joystick_count = pygame.joystick.get_count()
    if joystick_count == 0:
        print("No joysticks detected!")
        print("Make sure your joystick is connected and try again.")
        sys.exit(1)
    
    print(f"Found {joystick_count} joystick(s)\n")
    
    # List all joysticks
    for i in range(joystick_count):
        temp_joy = pygame.joystick.Joystick(i)
        temp_joy.init()
        print(f"Joystick {i}: {temp_joy.get_name()}")
        print(f"  Axes: {temp_joy.get_numaxes()}")
        print(f"  Buttons: {temp_joy.get_numbuttons()}")
        print(f"  Hats: {temp_joy.get_numhats()}")
        print(f"  Balls: {temp_joy.get_numballs()}")
        temp_joy.quit()
    
    # Use first joystick
    joystick_id = 0
    if joystick_count > 1:
        try:
            joystick_id = int(input(f"\nSelect joystick (0-{joystick_count-1}): "))
            if joystick_id < 0 or joystick_id >= joystick_count:
                joystick_id = 0
        except (ValueError, KeyboardInterrupt):
            joystick_id = 0
    
    joystick = pygame.joystick.Joystick(joystick_id)
    joystick.init()
    
    print(f"\nUsing joystick {joystick_id}: {joystick.get_name()}")
    print("=" * 80)
    print("Move sticks, press buttons, and pull triggers to see their values.")
    print("When at rest, note the values to determine appropriate deadzone.")
    print("Press Ctrl+C to exit.")
    print("=" * 80)
    print()
    
    try:
        while True:
            # Process events (required for joystick input)
            pygame.event.pump()
            
            # Clear screen (works on most terminals)
            print("\033[2J\033[H", end="")  # ANSI escape codes to clear screen
            
            print(f"Joystick: {joystick.get_name()}")
            print("=" * 80)
            
            # Display axes
            num_axes = joystick.get_numaxes()
            print(f"\nAxes ({num_axes}):")
            for i in range(num_axes):
                value = joystick.get_axis(i)
                # Color code based on value
                if abs(value) < 0.1:
                    status = "✓ REST"  # At rest
                elif abs(value) > 0.9:
                    status = "⚠ MAX"   # Near maximum
                else:
                    status = "  ACTIVE"
                
                bar_length = int(abs(value) * 40)
                bar = "█" * bar_length + "░" * (40 - bar_length)
                sign = "+" if value >= 0 else "-"
                
                print(f"  Axis {i:2d}: {value:7.4f} [{sign}{bar}] {status}")
            
            # Display buttons
            num_buttons = joystick.get_numbuttons()
            print(f"\nButtons ({num_buttons}):")
            pressed_buttons = []
            for i in range(num_buttons):
                if joystick.get_button(i):
                    pressed_buttons.append(str(i))
            if pressed_buttons:
                print(f"  Pressed: {', '.join(pressed_buttons)}")
            else:
                print("  None pressed")
            
            # Display hats (D-pads)
            num_hats = joystick.get_numhats()
            if num_hats > 0:
                print(f"\nHats ({num_hats}):")
                for i in range(num_hats):
                    hat = joystick.get_hat(i)
                    if hat != (0, 0):
                        print(f"  Hat {i}: {hat}")
                    else:
                        print(f"  Hat {i}: (0, 0) - centered")
            
            # Deadzone recommendations
            print("\n" + "=" * 80)
            print("Deadzone Recommendations:")
            
            # Check for non-zero values at rest
            rest_values = []
            for i in range(num_axes):
                value = joystick.get_axis(i)
                if abs(value) > 0.001:  # Very small threshold
                    rest_values.append((i, value))
            
            if rest_values:
                print("  ⚠ Warning: Some axes have non-zero values at rest:")
                max_rest = 0.0
                for axis_id, value in rest_values:
                    abs_val = abs(value)
                    max_rest = max(max_rest, abs_val)
                    print(f"    Axis {axis_id}: {value:.4f} (deadzone should be > {abs_val:.4f})")
                print(f"\n  Recommended deadzone: {max_rest + 0.05:.2f} (add 5% margin)")
            else:
                print("  ✓ All axes appear to be at rest (values near 0.0)")
                print("  Recommended deadzone: 0.10 (10% default)")
            
            # Trigger-specific info
            if num_axes > 4:
                print("\n  Trigger Analysis:")
                if num_axes > 5:
                    right_trigger = joystick.get_axis(4)
                    left_trigger = joystick.get_axis(5)
                    print(f"    Right trigger (axis 4): {right_trigger:.4f}")
                    print(f"    Left trigger (axis 5): {left_trigger:.4f}")
                    
                    # Normalized values
                    right_norm = (right_trigger + 1.0) / 2.0
                    left_norm = (left_trigger + 1.0) / 2.0
                    print(f"    Normalized - Right: {right_norm:.4f}, Left: {left_norm:.4f}")
                    
                    if right_trigger < -0.8:
                        print("    ✓ Right trigger at rest (should be ~0.0 when normalized)")
                    if left_trigger < -0.8:
                        print("    ✓ Left trigger at rest (should be ~0.0 when normalized)")
            
            print("\n" + "=" * 80)
            print("Press Ctrl+C to exit")
            
            time.sleep(0.05)  # Update ~20 times per second
            
    except KeyboardInterrupt:
        print("\n\nExiting...")
    finally:
        joystick.quit()
        pygame.quit()


if __name__ == "__main__":
    main()
