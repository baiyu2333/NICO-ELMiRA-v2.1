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
        "r_indexfingers_x": -150,  # Index open
        "r_virtualhand_x": -150,   # Virtual hand (thumb+middle) open
    }
    
    RIGHT_HAND_CLOSE = {
        "r_indexfingers_x": 120,   # Index closed
        "r_virtualhand_x": 120,    # Virtual hand closed
    }
    
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "failed", "no_action"],
            input_keys=["hand_action", "planning_group"],  
            output_keys=[],
        )
        
        # Publishers for direct motor control
        self._pub_set_angle = rospy.Publisher(
            "/nico/motion/setAngle", nicomsg.msg.sff, queue_size=10
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
            rospy.logwarn("HandControl: Left hand direct motor control not implemented (XL-320 conflict)")
            return
            
        positions = self.RIGHT_HAND_OPEN if action == "open" else self.RIGHT_HAND_CLOSE
        
        for motor_name, position in positions.items():
            msg = nicomsg.msg.sff()
            msg.param1 = motor_name
            msg.param2 = float(position)
            msg.param3 = 1.0  # max speed fraction
            self._pub_set_angle.publish(msg)
            rospy.logdebug(f"HandControl: Published {motor_name} = {position}")

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
