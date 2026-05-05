#!/usr/bin/env python
import numpy as np

import smach
from smach_ros import ServiceState, MonitorState
from open_manipulator_msgs.srv import SetJointPosition
from open_manipulator_msgs.msg import JointPosition
from sensor_msgs.msg import JointState
import rospy

from utils.constants import filter_commandable_joints


class JointTrajectoryIterator(smach.Iterator):
    """Executes a sequence of joint movements."""

    def __init__(
        self,
        srv_topic_head,
        sub_topic_head,
        srv_topic_left,
        sub_topic_left,
        srv_topic_right,
        sub_topic_right,
    ):
        super(JointTrajectoryIterator, self).__init__(
            outcomes=["succeeded", "preempted", "aborted"],
            input_keys=[
                "joint_trajectory",
            ],
            output_keys=[],
            it=lambda: range(0, len(self.userdata.joint_trajectory)),
            it_label="trajectory_step",
            exhausted_outcome="succeeded",
        )
        with self:
            trajectory_sm = smach.StateMachine(
                outcomes=["succeeded", "preempted", "aborted", "next_pose"],
                input_keys=["joint_trajectory", "trajectory_step"],
            )
            with trajectory_sm:
                # parse next joint state
                @smach.cb_interface(
                    input_keys=["joint_trajectory", "trajectory_step"],
                    output_keys=[
                        "names_head",
                        "positions_head",
                        "names_left",
                        "positions_left",
                        "names_right",
                        "positions_right",
                        "hand_action",
                        "planning_group",
                    ],
                    outcomes=["succeeded"],
                )
                def trajectory_step_cb(userdata):
                    joint_states = userdata.joint_trajectory[userdata.trajectory_step]
                    if "head" in joint_states:
                        names, positions, dropped = filter_commandable_joints(
                            "head",
                            joint_states["head"]["names"],
                            joint_states["head"]["positions"],
                        )
                        userdata.names_head = names
                        userdata.positions_head = positions
                        if dropped:
                            rospy.logwarn(f"MoveRobot: Dropping non-commandable head joints: {dropped}")
                    else:
                        userdata.names_head = []
                        userdata.positions_head = []
                    if "l_arm" in joint_states:
                        names, positions, dropped = filter_commandable_joints(
                            "l_arm",
                            joint_states["l_arm"]["names"],
                            joint_states["l_arm"]["positions"],
                        )
                        userdata.names_left = names
                        userdata.positions_left = positions
                        if dropped:
                            rospy.logwarn(f"MoveRobot: Dropping non-commandable left arm joints: {dropped}")
                    else:
                        userdata.names_left = []
                        userdata.positions_left = []
                    if "r_arm" in joint_states:
                        names, positions, dropped = filter_commandable_joints(
                            "r_arm",
                            joint_states["r_arm"]["names"],
                            joint_states["r_arm"]["positions"],
                        )
                        userdata.names_right = names
                        userdata.positions_right = positions
                        if dropped:
                            rospy.logwarn(f"MoveRobot: Dropping non-commandable right arm joints: {dropped}")
                    else:
                        userdata.names_right = []
                        userdata.positions_right = []
                    
                    # Optional inline hand action
                    userdata.hand_action = joint_states.get("hand_action", None)
                    userdata.planning_group = joint_states.get("planning_group", "r_arm")
                    return "succeeded"

                smach.StateMachine.add(
                    "SET_JOINT_TARGETS",
                    smach.CBState(trajectory_step_cb),
                    {"succeeded": "MOVE_ROBOT"},
                )
                smach.StateMachine.add(
                    "MOVE_ROBOT",
                    MoveRobot(
                        srv_topic_head,
                        sub_topic_head,
                        srv_topic_left,
                        sub_topic_left,
                        srv_topic_right,
                        sub_topic_right,
                    ),
                    transitions={
                        "movement_done": "CHECK_HAND_ACTION",
                        "aborted": "aborted"
                    },
                )

                # Check if we need to do an inline hand action after moving
                @smach.cb_interface(
                    input_keys=["hand_action"], 
                    outcomes=["execute_hand", "skip_hand"]
                )
                def check_hand_cb(userdata):
                    return "execute_hand" if userdata.hand_action else "skip_hand"

                smach.StateMachine.add(
                    "CHECK_HAND_ACTION",
                    smach.CBState(check_hand_cb),
                    {"execute_hand": "EXECUTE_HAND_ACTION", "skip_hand": "next_pose"},
                )
                
                from states.hand_control import HandControl
                smach.StateMachine.add(
                    "EXECUTE_HAND_ACTION",
                    HandControl(),
                    {
                        "succeeded": "next_pose", 
                        "failed": "next_pose", 
                        "no_action": "next_pose"
                    },
                )

            # close trajectory_sm
            smach.Iterator.set_contained_state(
                "EXECUTE_JOINT_TRAJECTORY", trajectory_sm, loop_outcomes=["next_pose"]
            )


class MoveRobot(smach.Concurrence):
    """Move robot head and arms in parallel.
    
    NOTE: Left hand (wrist/fingers) is non-functional, but left arm
    (shoulder/elbow) works and can be used for pointing/pushing.
    """

    def __init__(
        self,
        srv_topic_head,
        sub_topic_head,
        srv_topic_left,
        sub_topic_left,
        srv_topic_right,
        sub_topic_right,
    ):
        super(MoveRobot, self).__init__(
            input_keys=[
                "names_head",
                "positions_head",
                "names_left",
                "positions_left",
                "names_right",
                "positions_right",
            ],
            outcomes=["movement_done", "aborted"],
            default_outcome="aborted",  # Safely abort if something isn't completely successful
            child_termination_cb=lambda so: True if "aborted" in so.values() else False,
            outcome_map={
                "movement_done": {
                    "MOVE_HEAD": "succeeded",
                    "MOVE_LEFT_ARM": "succeeded",
                    "MOVE_RIGHT_ARM": "succeeded",
                },
            },
        )
        # Open the container
        with self:
            smach.Concurrence.add(
                "MOVE_HEAD",
                MoveRobotPart(srv_topic_head, sub_topic_head),
                remapping={"names": "names_head", "positions": "positions_head"},
            )
            smach.Concurrence.add(
                "MOVE_LEFT_ARM",
                MoveRobotPart(srv_topic_left, sub_topic_left),
                remapping={"names": "names_left", "positions": "positions_left"},
            )
            smach.Concurrence.add(
                "MOVE_RIGHT_ARM",
                MoveRobotPart(srv_topic_right, sub_topic_right),
                remapping={"names": "names_right", "positions": "positions_right"},
            )


class MoveRobotPart(smach.Sequence):
    """Move robot part and monitor state until success."""

    def __init__(self, srv_topic, sub_topic):
        super(MoveRobotPart, self).__init__(
            input_keys=["names", "positions"],
            outcomes=["succeeded", "aborted", "preempted"],
            connector_outcome="succeeded",
        )
        # Open the container
        with self:

            @smach.cb_interface(input_keys=["names", "positions"])
            def set_joint_position_request_cb(userdata, request):
                joint_position = JointPosition()
                joint_position.joint_name = userdata.names
                joint_position.position = userdata.positions
                request.joint_position = joint_position
                return request

            smach.Sequence.add(
                "START_MOVEMENT",
                ServiceState(
                    srv_topic,
                    SetJointPosition,
                    request_cb=set_joint_position_request_cb,
                    input_keys=["names", "positions"],
                ),
            )

            # TODO precision input parameter?
            @smach.cb_interface(input_keys=["names", "positions"])
            def target_joint_state_reached_cb(userdata, message):
                if not userdata.names:
                    return False

                missing = [name for name in userdata.names if name not in message.name]
                if missing:
                    rospy.logwarn(f"MoveRobotPart: Missing joints in feedback: {missing}")
                    return False

                joint_ids = np.argsort(message.name)
                selected_ids = joint_ids[
                    np.searchsorted(message.name, userdata.names, sorter=joint_ids)
                ]
                ordered_state = np.array(message.position)[selected_ids]
                if len(message.velocity) == len(message.name):
                    ordered_velocity = np.array(message.velocity)[selected_ids]
                else:
                    ordered_velocity = np.zeros(len(userdata.names))
                return not (
                    np.allclose(userdata.positions, ordered_state, atol=0.052)  # < ~3°
                    and np.all(np.abs(ordered_velocity) < 0.01)
                )

            smach.Sequence.add(
                "WAIT_UNTIL_REACHED",
                MonitorState(
                    sub_topic,
                    JointState,
                    target_joint_state_reached_cb,
                    max_checks=1500,  # publisher at 50hz = 30 second timeout
                    input_keys=["names", "positions"],
                ),
                transitions={"valid": "succeeded", "invalid": "succeeded"},
            )
