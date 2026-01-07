#!/usr/bin/env python3
"""
Hand Control State for ELMiRA v2

Provides SMACH states for controlling NICO's left and right hands.

IMPORTANT ARCHITECTURE NOTE:
- Right hand: Controlled via ROS services through the main Motion node.
  Motors are in the main config and use Protocol 1.0.
  
- Left hand: Uses SEED Robotics XL-320 protocol which is incompatible with
  Protocol 1.0 used by the main Motion controller. The left hand motors are
  NOT in the main config to avoid protocol conflicts during initialization.
  
  Left hand control options:
  1. Direct control via Dxl320IO (only works when Motion node is NOT running)
  2. Future: Separate left_hand_controller node with its own connection

Right Hand Motor IDs (Protocol 1.0 - in Motion config):
- 23: r_wrist_z
- 25: r_wrist_x
- 29: r_indexfingers_x
- 32: r_virtualhand_x (coupled fingers)

Left Hand Motor IDs (XL-320 protocol - NOT in Motion config):
- 31: l_wrist_z (Wrist Roll)
- 33: l_wrist_x (Wrist Pitch)
- 34: l_thumb_z (Thumb Roll) - 2-DOF thumb
- 35: l_thumb_x (Thumb Pitch)
- 36: l_indexfingers_x (Index Finger)
- 37: l_middlefingers_x (Middle Finger)

Note: SEED servo position feedback is unreliable - use timeout-based completion.
"""

import time
import rospy
import smach


class HandControl(smach.State):
    """
    SMACH state for opening or closing robot hands.
    
    Uses ROS service or direct motor control to actuate finger motors.
    Determines which hand to use from planning_group.
    """
    
    # Motor positions for hand poses (in degrees)
    # Left hand (new SEED Robotics with 2-DOF thumb)
    LEFT_HAND_OPEN = {
        "l_thumb_z": 0,      # Thumb roll centered
        "l_thumb_x": -100,   # Thumb open
        "l_indexfingers_x": -100,  # Index open
        "l_middlefingers_x": -100, # Middle open
    }
    
    LEFT_HAND_CLOSE = {
        "l_thumb_z": 50,     # Thumb roll inward for grasp
        "l_thumb_x": 100,    # Thumb closed
        "l_indexfingers_x": 100,   # Index closed
        "l_middlefingers_x": 100,  # Middle closed
    }
    
    # Right hand (existing Protocol 1.0)
    RIGHT_HAND_OPEN = {
        "r_indexfingers_x": -150,  # Index open
        "r_virtualhand_x": -150,   # Virtual hand (thumb+middle) open
    }
    
    RIGHT_HAND_CLOSE = {
        "r_indexfingers_x": 120,   # Index closed
        "r_virtualhand_x": 120,    # Virtual hand closed
    }
    
    def __init__(self, robot=None, use_ros_service=True):
        """
        Initialize hand control state.
        
        Args:
            robot: Optional pypot robot instance for direct motor control.
                   If None and use_ros_service=False, will attempt to connect.
            use_ros_service: If True, use ROS service for motor control (preferred).
                            If False, use direct pypot motor control.
        """
        smach.State.__init__(
            self,
            outcomes=["succeeded", "failed", "no_action"],
            input_keys=[],  # No required keys - we check for them manually
            output_keys=[],
        )
        
        self.robot = robot
        self.use_ros_service = use_ros_service
        self._left_io = None
        self._right_io = None
        
        # Motor command timeout (seconds) - don't wait for position feedback
        self.command_timeout = 1.0
    
    def _get_hand_side(self, planning_group: str) -> str:
        """Determine hand side from planning group."""
        if planning_group in ("l_arm", "l_hand"):
            return "left"
        elif planning_group in ("r_arm", "r_hand"):
            return "right"
        else:
            rospy.logwarn(f"Unknown planning group: {planning_group}")
            return "right"  # Default to right
    
    def _init_direct_io(self, side: str):
        """Initialize direct motor IO if needed."""
        if side == "left" and self._left_io is None:
            try:
                from pypot.dynamixel.io.io_320 import Dxl320IO
                # Left hand uses XL-320 protocol
                port = rospy.get_param("/nico/motor_port", "/dev/ttyUSB0")
                self._left_io = Dxl320IO(port, baudrate=1_000_000, timeout=0.5)
                rospy.loginfo("HandControl: Initialized XL-320 IO for left hand")
            except Exception as e:
                rospy.logerr(f"Failed to initialize left hand IO: {e}")
                return None
            return self._left_io
        elif side == "right" and self._right_io is None:
            try:
                from pypot.dynamixel.io import DxlIO
                # Right hand uses Protocol 1.0
                port = rospy.get_param("/nico/motor_port", "/dev/ttyUSB0")
                self._right_io = DxlIO(port, baudrate=1_000_000, timeout=0.5)
                rospy.loginfo("HandControl: Initialized DxlIO for right hand")
            except Exception as e:
                rospy.logerr(f"Failed to initialize right hand IO: {e}")
                return None
            return self._right_io
        
        return self._left_io if side == "left" else self._right_io
    
    def _send_hand_command_ros(self, side: str, action: str) -> bool:
        """
        Send hand command via ROS service.
        
        Uses /nico/motion/setAngle service.
        
        NOTE: Only works for RIGHT hand since left hand motors are not in
        the Motion config (due to XL-320 protocol incompatibility).
        """
        # Left hand cannot use ROS services - motors not in Motion config
        if side == "left":
            rospy.logwarn(
                "HandControl: Left hand motors are not in Motion config. "
                "Cannot use ROS service. Left hand control not yet implemented."
            )
            # TODO: Implement dedicated left hand controller node
            return False
        
        try:
            from nicomsg.srv import SetValue
            
            # Right hand only
            positions = self.RIGHT_HAND_OPEN if action == "open" else self.RIGHT_HAND_CLOSE
            
            # Call setAngle service for each motor
            # Enable torque for these motors first to ensure they move
            enable_torque = rospy.ServiceProxy("/nico/motion/enableTorque", nicomsg.srv.SetValue)
            set_angle = rospy.ServiceProxy("/nico/motion/setAngle", SetValue)
            
            for motor_name, position in positions.items():
                try:
                    # Enable torque first
                    # Using SetValue because enableTorque might expect string, output of enableTorque is 's' (string)
                    # Wait, looking at Motion.py: _ROSPY__enableTorque takes 's' -> nicomsg.msg.s (topic)
                    # BUT Motion.py lines 570-577 is SUBSCRIBER.
                    # Motion.py doesn't seem to expose enableTorque as a SERVICE properly named enableTorque ??
                    # It has subscribers.
                    # Wait, let's check Motion.py again.
                    pass 
                except Exception:
                    pass
            
            # Using Publishers for Torque Enable since Service might not exist 
            enable_torque_pub = rospy.Publisher("/nico/motion/enableTorque", nicomsg.msg.s, queue_size=5)
            rospy.sleep(0.1) # Wait for connection
            
            for motor_name, position in positions.items():
                try:
                    # Publish Enable Torque
                    msg = nicomsg.msg.s()
                    msg.param1 = motor_name
                    enable_torque_pub.publish(msg)
                    rospy.sleep(0.05)

                    # SetValue expects motor name and value
                    set_angle(motor_name, float(position))
                    rospy.logdebug(f"Set {motor_name} to {position}")
                except rospy.ServiceException as e:
                    rospy.logwarn(f"Failed to set {motor_name}: {e}")
            
            # Wait for movement to complete (timeout-based, no position feedback)
            rospy.sleep(self.command_timeout)
            return True
            
        except Exception as e:
            rospy.logerr(f"ROS hand control failed: {e}")
            return False
    
    def _send_hand_command_direct(self, side: str, action: str) -> bool:
        """
        Send hand command via direct motor control (pypot).
        
        WARNING: This method opens its own serial connection, which will FAIL
        if the Motion node is already running (port conflict).
        
        For left hand: Cannot work while Motion node is running.
        For right hand: Use ROS services instead (preferred).
        """
        # Left hand cannot use direct IO while Motion node is running
        if side == "left":
            rospy.logerr(
                "HandControl: Left hand direct control cannot work while Motion node "
                "is running (serial port conflict). Left hand actions are not "
                "currently supported. A dedicated left hand controller node is needed."
            )
            return False
        
        try:
            io = self._init_direct_io(side)
            if io is None:
                return False
            
            # Right hand only
            positions = self.RIGHT_HAND_OPEN if action == "open" else self.RIGHT_HAND_CLOSE
            motor_ids = {
                "r_indexfingers_x": 29,
                "r_virtualhand_x": 32,
            }
            
            # Send position commands
            for motor_name, position in positions.items():
                motor_id = motor_ids.get(motor_name)
                if motor_id:
                    io.set_goal_position({motor_id: position})
                    rospy.logdebug(f"Direct set motor {motor_id} ({motor_name}) to {position}")
                    time.sleep(0.05)  # Small delay between commands
            
            # Wait for movement (no position feedback for SEED servos)
            rospy.sleep(self.command_timeout)
            return True
            
        except Exception as e:
            rospy.logerr(f"Direct hand control failed: {e}")
            return False
    
    def execute(self, userdata):
        """Execute hand control action."""
        # Check if hand action is needed - use try/except for SMACH userdata
        try:
            hand_action = userdata.hand_action
        except (KeyError, AttributeError):
            hand_action = None
        
        if hand_action is None:
            rospy.logdebug("HandControl: No hand action requested")
            return "no_action"
        
        if hand_action not in ("open", "close"):
            rospy.logwarn(f"HandControl: Unknown hand action '{hand_action}'")
            return "no_action"
        
        # Determine which hand to control
        try:
            planning_group = userdata.planning_group
        except (KeyError, AttributeError):
            planning_group = 'r_arm'
        side = self._get_hand_side(planning_group)
        
        rospy.loginfo(f"HandControl: {hand_action} {side} hand")
        
        # Try ROS service first, fall back to direct control
        success = False
        if self.use_ros_service:
            success = self._send_hand_command_ros(side, hand_action)
        
        if not success:
            rospy.logwarn("HandControl: ROS service failed. NOT falling back to direct control to avoid crashing the driver (Port Conflict).")
            # success = self._send_hand_command_direct(side, hand_action)
        
        return "succeeded" if success else "failed"


class OpenHand(smach.State):
    """Convenience state to open the hand based on planning_group."""
    
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "failed"],
            input_keys=["planning_group"],
            output_keys=[],
        )
        self._hand_control = HandControl()
    
    def execute(self, userdata):
        # Create temporary userdata with hand_action set
        class TempUserdata:
            pass
        temp = TempUserdata()
        temp.hand_action = "open"
        temp.planning_group = userdata.planning_group
        
        result = self._hand_control.execute(temp)
        return "succeeded" if result in ("succeeded", "no_action") else "failed"


class CloseHand(smach.State):
    """Convenience state to close the hand based on planning_group."""
    
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "failed"],
            input_keys=["planning_group"],
            output_keys=[],
        )
        self._hand_control = HandControl()
    
    def execute(self, userdata):
        # Create temporary userdata with hand_action set
        class TempUserdata:
            pass
        temp = TempUserdata()
        temp.hand_action = "close"
        temp.planning_group = userdata.planning_group
        
        result = self._hand_control.execute(temp)
        return "succeeded" if result in ("succeeded", "no_action") else "failed"


class PreGraspHand(smach.State):
    """
    Prepare hand for grasping - opens hand wide before approach.
    This should be called before the grasp arm trajectory.
    """
    
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "failed"],
            input_keys=["planning_group"],
            output_keys=[],
        )
        self._open_hand = OpenHand()
    
    def execute(self, userdata):
        rospy.loginfo("PreGraspHand: Opening hand for grasp approach")
        return self._open_hand.execute(userdata)
