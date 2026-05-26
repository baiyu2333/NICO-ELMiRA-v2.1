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
import re
import rospy
import smach
import nicomsg.msg

from utils.constants import palm_sensor_available


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
        "r_indexfingers_x": 120,   # Current motor config only exposes this right-hand joint.
    }
    
    RIGHT_HAND_CLOSE = {
        "r_indexfingers_x": -150,
    }
    
    # Left hand XL-320 motor IDs (Protocol 2.0).
    # A safe ping scan on this robot found responders at:
    # 30, 31, 33, 34, 35, 36, 37.
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
    LEFT_FINGER_IDS = [33, 34, 35, 36, 37]
    
    # Left wrist motor IDs (also XL-320). The normal arm trajectory cannot
    # command these through pypot, so keep them aligned here when the hand acts.
    LEFT_WRIST_Z_ID = 30
    LEFT_WRIST_X_ID = 31
    LEFT_WRIST_Z_DEG = 80.0   # Match ELMiRA's +1.39 rad palm-facing pose.
    LEFT_WRIST_X_DEG = 0.0
    
    # Positions for XL-320 left hand
    LEFT_HAND_OPEN_POS = -150.0    # All fingers fully open
    LEFT_HAND_CLOSE_POS = 60.0     # All fingers closed for grasping

    # Current lab hardware has the old XL-320 left hand mounted as the
    # physical right hand. Keep these values param-driven so they can be
    # changed from the launch file without touching code.
    # ID 31: forearm rotation; ID 33: wrist pitch.
    # Fingers: 34 thumb base, 35 thumb MCP/IP, 36 index,
    # 37 middle and ring together.
    RIGHT_XL320_FINGER_IDS = [34, 35, 36, 37]
    RIGHT_XL320_WRIST_Z_ID = 31
    RIGHT_XL320_WRIST_X_ID = 33
    RIGHT_TOUCH_WRIST_Z_DEG = 80.0
    RIGHT_TOUCH_WRIST_X_DEG = -45.0
    RIGHT_TOUCH_HOLD_SEC = 2.0
    RIGHT_HAND_OPEN_POS = -150.0
    RIGHT_HAND_CLOSE_POS = 60.0

    @staticmethod
    def _parse_int_list(value, default):
        if value is None:
            return list(default)
        if isinstance(value, str):
            parsed = [int(match) for match in re.findall(r"-?\d+", value)]
        else:
            parsed = [int(item) for item in value]
        return parsed or list(default)

    @staticmethod
    def _parse_bool(value, default=False):
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _deg_to_xl320_raw(deg: float) -> int:
        raw = int((float(deg) + 150.0) / 300.0 * 1023.0)
        return max(0, min(1023, raw))
    
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

        self._left_finger_ids = self._parse_int_list(
            rospy.get_param("/elmira/left_xl320_finger_ids", self.LEFT_FINGER_IDS),
            self.LEFT_FINGER_IDS,
        )
        self._left_wrist_z_id = int(
            rospy.get_param("/elmira/left_xl320_wrist_z_id", self.LEFT_WRIST_Z_ID)
        )
        self._left_wrist_x_id = int(
            rospy.get_param("/elmira/left_xl320_wrist_x_id", self.LEFT_WRIST_X_ID)
        )
        self._left_hand_open_pos = float(
            rospy.get_param("/elmira/left_hand_open_deg", self.LEFT_HAND_OPEN_POS)
        )
        self._left_hand_close_pos = float(
            rospy.get_param("/elmira/left_hand_close_deg", self.LEFT_HAND_CLOSE_POS)
        )
        self._left_wrist_positions = {
            self._left_wrist_z_id: float(
                rospy.get_param("/elmira/left_wrist_z_deg", self.LEFT_WRIST_Z_DEG)
            ),
            self._left_wrist_x_id: float(
                rospy.get_param("/elmira/left_wrist_x_deg", self.LEFT_WRIST_X_DEG)
            ),
        }
        rospy.loginfo(
            "HandControl: left XL-320 config wrist=%s fingers=%s open=%.1f close=%.1f",
            self._left_wrist_positions,
            self._left_finger_ids,
            self._left_hand_open_pos,
            self._left_hand_close_pos,
        )

        self._refresh_right_xl320_params()
        rospy.loginfo(
            "HandControl: swapped right XL-320 enabled=%s wrist_touch=%s fingers=%s open=%.1f close=%.1f hold=%.1fs",
            self._use_swapped_right_xl320_hand,
            self._right_touch_wrist_positions,
            self._right_finger_ids,
            self._right_hand_open_pos,
            self._right_hand_close_pos,
            self._right_touch_hold_sec,
        )
        
        # Palm sensor monitor
        self._palm = get_palm_monitor()

    def _refresh_right_xl320_params(self):
        """Reload current physical right-hand XL-320 params from ROS."""
        self._use_swapped_right_xl320_hand = self._parse_bool(
            rospy.get_param("/elmira/use_swapped_right_xl320_hand", True),
            True,
        )
        self._right_finger_ids = self._parse_int_list(
            rospy.get_param("/elmira/right_xl320_finger_ids", self.RIGHT_XL320_FINGER_IDS),
            self.RIGHT_XL320_FINGER_IDS,
        )
        self._right_wrist_z_id = int(
            rospy.get_param("/elmira/right_xl320_wrist_z_id", self.RIGHT_XL320_WRIST_Z_ID)
        )
        self._right_wrist_x_id = int(
            rospy.get_param("/elmira/right_xl320_wrist_x_id", self.RIGHT_XL320_WRIST_X_ID)
        )
        self._right_hand_open_pos = float(
            rospy.get_param("/elmira/right_hand_open_deg", self.RIGHT_HAND_OPEN_POS)
        )
        self._right_hand_close_pos = float(
            rospy.get_param("/elmira/right_hand_close_deg", self.RIGHT_HAND_CLOSE_POS)
        )
        self._right_touch_hold_sec = float(
            rospy.get_param("/elmira/right_touch_hold_sec", self.RIGHT_TOUCH_HOLD_SEC)
        )
        self._right_touch_wrist_positions = {
            self._right_wrist_z_id: float(
                rospy.get_param("/elmira/right_touch_wrist_z_deg", self.RIGHT_TOUCH_WRIST_Z_DEG)
            ),
            self._right_wrist_x_id: float(
                rospy.get_param("/elmira/right_touch_wrist_x_deg", self.RIGHT_TOUCH_WRIST_X_DEG)
            ),
        }
    
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
        if side == "right" and self._use_swapped_right_xl320_hand:
            self._move_right_hand_xl320(action)
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
        target_pos = (
            self._left_hand_open_pos if action == "open" else self._left_hand_close_pos
        )
        
        # Convert degrees to XL-320 raw value (0-1023, center=512)
        raw_pos = int((target_pos + 150.0) / 300.0 * 1023.0)
        raw_pos = max(0, min(1023, raw_pos))
        
        try:
            rospy.loginfo(f"HandControl: XL-320 left hand {action} via ROS topic...")
            
            # ALL left hand XL-320 motor IDs (wrist + fingers)
            all_motor_ids = sorted(
                set(list(self._left_wrist_positions.keys()) + self._left_finger_ids)
            )
            
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
            for wrist_id, wrist_deg in self._left_wrist_positions.items():
                raw_wrist = int((wrist_deg + 150.0) / 300.0 * 1023.0)
                raw_wrist = max(0, min(1023, raw_wrist))
                self._send_xl320_cmd(wrist_id, 30, raw_wrist)
                rospy.sleep(0.005)
            rospy.loginfo(
                f"HandControl: Left wrist aligned for grasp {self._left_wrist_positions}"
            )
            
            # Step 4: Set finger positions
            for motor_id in self._left_finger_ids:
                self._send_xl320_cmd(motor_id, 30, raw_pos)
                rospy.sleep(0.005)
            
            rospy.loginfo(
                f"HandControl: Left hand XL-320 {action} -> "
                f"fingers {self._left_finger_ids} to {target_pos}° (raw={raw_pos})"
            )
        except Exception as e:
            rospy.logerr(f"HandControl: XL-320 left hand {action} error: {e}")

    def _move_right_hand_xl320(self, action: str):
        """Control the swapped physical right hand via XL-320 Protocol 2.0."""
        self._refresh_right_xl320_params()
        target_pos = (
            self._right_hand_open_pos if action == "open" else self._right_hand_close_pos
        )
        raw_pos = self._deg_to_xl320_raw(target_pos)

        try:
            rospy.loginfo(f"HandControl: XL-320 swapped right hand {action} via ROS topic...")
            all_motor_ids = sorted(
                set(list(self._right_touch_wrist_positions.keys()) + self._right_finger_ids)
            )
            for motor_id in all_motor_ids:
                self._send_xl320_cmd(motor_id, 24, 1)
                rospy.sleep(0.005)
            for motor_id in all_motor_ids:
                self._send_xl320_cmd(motor_id, 32, 150)
                rospy.sleep(0.005)
            self._align_right_touch_wrist_xl320()
            for motor_id in self._right_finger_ids:
                self._send_xl320_cmd(motor_id, 30, raw_pos)
                rospy.sleep(0.005)
            rospy.loginfo(
                f"HandControl: XL-320 swapped right hand {action} -> "
                f"fingers {self._right_finger_ids} to {target_pos} deg (raw={raw_pos})"
            )
        except Exception as e:
            rospy.logerr(f"HandControl: XL-320 swapped right hand {action} error: {e}")

    def _align_right_touch_wrist_xl320(self):
        """Align swapped physical right wrist downward for table-object touch."""
        self._refresh_right_xl320_params()
        if not self._use_swapped_right_xl320_hand:
            rospy.loginfo("HandControl: swapped right XL-320 hand disabled; skipping touch wrist alignment")
            return
        all_motor_ids = sorted(self._right_touch_wrist_positions.keys())
        for motor_id in all_motor_ids:
            self._send_xl320_cmd(motor_id, 24, 1)
            rospy.sleep(0.005)
        for motor_id in all_motor_ids:
            self._send_xl320_cmd(motor_id, 32, 120)
            rospy.sleep(0.005)
        commanded = {}
        for wrist_id, wrist_deg in self._right_touch_wrist_positions.items():
            raw_wrist = self._deg_to_xl320_raw(wrist_deg)
            self._send_xl320_cmd(wrist_id, 30, raw_wrist)
            commanded[wrist_id] = {"deg": wrist_deg, "raw": raw_wrist}
            rospy.sleep(0.005)
        rospy.loginfo(
            "HandControl: swapped right wrist touch alignment %s hold=%.1fs",
            commanded,
            self._right_touch_hold_sec,
        )
    
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
            if not palm_sensor_available(side):
                rospy.loginfo(
                    f"HandControl: {side} hand command completed; palm sensor verification unavailable"
                )
                return True

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
        
        if hand_action not in ("open", "close", "touch_wrist"):
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
            if hand_action == "touch_wrist":
                if side == "right":
                    self._align_right_touch_wrist_xl320()
                    rospy.sleep(max(0.0, self._right_touch_hold_sec))
                    success = True
                else:
                    rospy.loginfo("HandControl: touch_wrist is only configured for the current right hand")
                    success = True
            elif hand_action == "open":
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
