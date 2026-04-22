#!/usr/bin/env python3
"""
Hand Control State for ELMiRA v2

Provides SMACH states for controlling NICO's left and right hands
via the Motion node's built-in openHand/closeHand ROS topics,
with palm sensor feedback to confirm grasps.

Architecture:
- Right hand: Controlled via /nico/motion/openHand and /nico/motion/closeHand
  topics (nicomsg.msg.s) through the main Motion node.
- Left hand: Same topics, but left hand hardware may be non-functional
  depending on the NICO variant.

Palm Sensor Feedback:
- /nico/motion/palm_sensor/right (nicomsg.msg.i) — higher value = more pressure
- /nico/motion/palm_sensor/left (nicomsg.msg.i)
- Used to confirm when an object is grasped (reading above threshold).
"""

import time
import threading
import rospy
import smach
import nicomsg.msg


# Palm sensor threshold to consider an object "grasped"
# Typical reading: ~0 when empty, ~200-1000+ when grasping an object
PALM_SENSOR_GRASP_THRESHOLD = 100

# Timeout for waiting for grasp confirmation (seconds)
GRASP_CONFIRM_TIMEOUT = 3.0

# Time to wait for hand movement to complete (seconds)
HAND_MOVE_DURATION = 1.5


class PalmSensorMonitor:
    """Monitors palm sensor readings from the Motion node."""
    
    def __init__(self):
        self._left_reading = 0
        self._right_reading = 0
        self._lock = threading.Lock()
        
        # Subscribe to palm sensor topics
        self._sub_left = rospy.Subscriber(
            "/nico/motion/palm_sensor/left",
            nicomsg.msg.i,
            self._left_cb,
        )
        self._sub_right = rospy.Subscriber(
            "/nico/motion/palm_sensor/right",
            nicomsg.msg.i,
            self._right_cb,
        )
        rospy.loginfo("PalmSensorMonitor: Subscribed to palm sensor topics")
    
    def _left_cb(self, msg):
        with self._lock:
            self._left_reading = msg.param1
    
    def _right_cb(self, msg):
        with self._lock:
            self._right_reading = msg.param1
    
    def get_reading(self, side: str) -> int:
        """Get current palm sensor reading for the given side."""
        with self._lock:
            return self._left_reading if side == "left" else self._right_reading
    
    def is_grasping(self, side: str, threshold: int = PALM_SENSOR_GRASP_THRESHOLD) -> bool:
        """Check if the palm sensor indicates an object is grasped."""
        return self.get_reading(side) > threshold
    
    def wait_for_grasp(self, side: str, timeout: float = GRASP_CONFIRM_TIMEOUT,
                       threshold: int = PALM_SENSOR_GRASP_THRESHOLD) -> bool:
        """
        Wait until palm sensor detects a grasp or timeout.
        
        Returns True if grasp was detected, False if timed out.
        """
        start = time.time()
        rate = rospy.Rate(20)  # Check at 20Hz
        while (time.time() - start) < timeout and not rospy.is_shutdown():
            if self.is_grasping(side, threshold):
                reading = self.get_reading(side)
                rospy.loginfo(
                    f"PalmSensor: Grasp confirmed on {side} hand "
                    f"(reading={reading}, threshold={threshold})"
                )
                return True
            rate.sleep()
        
        reading = self.get_reading(side)
        rospy.logwarn(
            f"PalmSensor: Grasp NOT confirmed on {side} hand after {timeout}s "
            f"(reading={reading}, threshold={threshold})"
        )
        return False


# Singleton palm sensor monitor (shared across all hand control states)
_palm_monitor = None

def get_palm_monitor() -> PalmSensorMonitor:
    """Get or create the singleton PalmSensorMonitor."""
    global _palm_monitor
    if _palm_monitor is None:
        _palm_monitor = PalmSensorMonitor()
    return _palm_monitor


class HandControl(smach.State):
    """
    SMACH state for opening or closing robot hands.
    
    Uses the Motion node's built-in openHand/closeHand ROS topics.
    Integrates palm sensor feedback for grasp verification.
    """
    
    # Right hand positions (in degrees)
    RIGHT_HAND_OPEN = {
        "r_indexfingers_x": 120,   # Index open
        "r_virtualhand_x": 120,    # Virtual hand (thumb+middle) open
    }
    
    RIGHT_HAND_CLOSE = {
        "r_indexfingers_x": -150,  # Index closed
        "r_virtualhand_x": -150,   # Virtual hand closed
    }
    
    # Left hand XL-320 motor IDs (Protocol 2.0)
    LEFT_HAND_MOTOR_IDS = {
        "l_wrist_z":          30,   # Forearm rotation
        "l_wrist_x":          31,   # Wrist flex  
        "l_thumb_x":          33,   # Thumb
        "l_indexfinger_x":    34,   # Index finger
        "l_middlefingers_x":  35,   # Middle finger
        "l_ring_x":           36,   # Ring finger (coupled with middle)
        "l_pinky_x":          37,   # Pinky (coupled with middle)
    }
    
    # Left hand finger IDs only (for open/close, excluding wrist)
    # 34: Thumb base, 35: Thumb joint, 36: Index, 37: Other fingers
    LEFT_FINGER_IDS = [34, 35, 36, 37]
    
    # Left wrist motor IDs (also XL-320)
    LEFT_WRIST_IDS = {
        31: 0.0,    # 小臂腕 (forearm rotation) -> neutral
        33: 0.0,    # 手腕 (wrist flex) -> neutral
    }
    
    # Positions for XL-320 left hand
    LEFT_HAND_OPEN_POS = -150.0    # All fingers fully open
    LEFT_HAND_CLOSE_POS = 60.0     # All fingers closed for grasping
    
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "failed", "no_action"],
            input_keys=["hand_action", "planning_group"],  
            output_keys=[],
        )
        
        # Publishers for direct motor control (right hand Protocol 1.0)
        self._pub_set_angle = rospy.Publisher(
            "/nico/motion/setAngle", nicomsg.msg.sff, queue_size=10
        )
        
        # Publisher for XL-320 left hand commands (Protocol 2.0 via Motion node)
        self._pub_xl320 = rospy.Publisher(
            "/nico/motion/xl320_cmd", nicomsg.msg.sff, queue_size=10
        )
        
        # Palm sensor monitor
        self._palm = get_palm_monitor()
    
    def _get_hand_side(self, planning_group: str) -> str:
        """Determine hand side from planning group."""
        if planning_group in ("l_arm", "l_hand"):
            return "left"
        elif planning_group in ("r_arm", "r_hand"):
            return "right"
        else:
            rospy.logwarn(f"HandControl: Unknown planning group: {planning_group}")
            return "right"  # Default to right
    
    def _move_hand_motors(self, side: str, action: str):
        """Send direct motor commands for the hand."""
        if side == "left":
            self._move_left_hand_xl320(action)
            return
            
        positions = self.RIGHT_HAND_OPEN if action == "open" else self.RIGHT_HAND_CLOSE
        
        for motor_name, position in positions.items():
            msg = nicomsg.msg.sff()
            msg.param1 = motor_name
            msg.param2 = float(position)
            msg.param3 = 1.0  # max speed fraction
            self._pub_set_angle.publish(msg)
            rospy.logdebug(f"HandControl: Published {motor_name} = {position}")
            rospy.sleep(0.1)
    
    def _move_left_hand_xl320(self, action: str):
        """Control the left hand via XL-320 Protocol 2.0.
        
        Sends commands through the /nico/motion/xl320_cmd ROS topic,
        which is handled by the Motion node using its own serial connection
        and pypot lock to safely send Protocol 2.0 packets.
        """
        target_pos = self.LEFT_HAND_OPEN_POS if action == "open" else self.LEFT_HAND_CLOSE_POS
        
        # Convert degrees to XL-320 raw value (0-1023, center=512)
        raw_pos = int((target_pos + 150.0) / 300.0 * 1023.0)
        raw_pos = max(0, min(1023, raw_pos))
        
        try:
            rospy.loginfo(f"HandControl: XL-320 left hand {action} via ROS topic...")
            
            # ALL left hand XL-320 motor IDs (wrist + fingers)
            all_motor_ids = list(self.LEFT_WRIST_IDS.keys()) + self.LEFT_FINGER_IDS
            
            # Step 1: Enable torque on ALL motors (register 24 = 1)
            for motor_id in all_motor_ids:
                self._send_xl320_cmd(motor_id, 24, 1)
                rospy.sleep(0.005)
            rospy.loginfo(f"HandControl: Torque enabled for XL-320 motors {all_motor_ids}")
            
            # Step 2: Set moving speed (register 32 = 150)
            for motor_id in all_motor_ids:
                self._send_xl320_cmd(motor_id, 32, 150)
                rospy.sleep(0.005)
            
            # Step 3: Set wrist positions (register 30 = goal position)
            for wrist_id, wrist_deg in self.LEFT_WRIST_IDS.items():
                raw_wrist = int((wrist_deg + 150.0) / 300.0 * 1023.0)
                raw_wrist = max(0, min(1023, raw_wrist))
                self._send_xl320_cmd(wrist_id, 30, raw_wrist)
                rospy.sleep(0.005)
            rospy.loginfo("HandControl: Left wrist set to neutral")
            
            # Step 4: Set finger positions
            for motor_id in self.LEFT_FINGER_IDS:
                self._send_xl320_cmd(motor_id, 30, raw_pos)
                rospy.sleep(0.005)
            
            rospy.loginfo(
                f"HandControl: Left hand XL-320 {action} -> "
                f"fingers {self.LEFT_FINGER_IDS} to {target_pos}° (raw={raw_pos})"
            )
        except Exception as e:
            rospy.logerr(f"HandControl: XL-320 left hand {action} error: {e}")
    
    def _send_xl320_cmd(self, motor_id: int, register: int, value: int):
        """Send a single XL-320 command via the Motion node's ROS topic."""
        msg = nicomsg.msg.sff()
        msg.param1 = str(motor_id)
        msg.param2 = float(register)
        msg.param3 = float(value)
        self._pub_xl320.publish(msg)
    

    def _open_hand(self, side: str) -> bool:
        """Open the specified hand."""
        rospy.loginfo(f"HandControl: Opening {side} hand")
        
        self._move_hand_motors(side, "open")
        
        # Wait for movement
        rospy.sleep(HAND_MOVE_DURATION)
        
        # Log palm sensor reading after opening
        reading = self._palm.get_reading(side)
        rospy.loginfo(f"HandControl: {side} hand opened (palm sensor: {reading})")
        return True
    
    def _close_hand(self, side: str, verify_grasp: bool = True) -> bool:
        """
        Close the specified hand.
        
        If verify_grasp is True, waits for palm sensor to confirm grasp.
        """
        rospy.loginfo(f"HandControl: Closing {side} hand")
        
        self._move_hand_motors(side, "close")
        
        # Wait for movement to complete
        rospy.sleep(HAND_MOVE_DURATION)
        
        if verify_grasp:
            # Check palm sensor for grasp confirmation
            grasped = self._palm.wait_for_grasp(side)
            if grasped:
                rospy.loginfo(f"HandControl: Object grasped by {side} hand!")
            else:
                rospy.logwarn(f"HandControl: {side} hand closed but no object detected by palm sensor")
            return True  # Still return True - the hand moved even if no object
        
        return True
    
    def execute(self, userdata):
        """Execute hand control action."""
        # Check if hand action is needed
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
            planning_group = "r_arm"
        side = self._get_hand_side(planning_group)
        
        rospy.loginfo(f"HandControl: {hand_action} {side} hand")
        
        try:
            if hand_action == "open":
                success = self._open_hand(side)
            else:
                success = self._close_hand(side, verify_grasp=True)
            
            return "succeeded" if success else "failed"
        except Exception as e:
            rospy.logerr(f"HandControl: Error during {hand_action}: {e}")
            return "failed"


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
