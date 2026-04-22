#!/usr/bin/env python3

import json
import rospy
import smach
import smach_ros

from actionlib_msgs.msg import GoalStatus
from elmira.msg import PerformASRAction
from elmira.srv import PromptTextLLM, PromptVisionLLM
from nicomsg.srv import SayText
from nicomsg.msg import empty
from std_msgs.msg import String

from utils.constants import LLM_CHAT_SERVICE, LLM_VISION_SERVICE
from utils.response_parser import parse_llm_actions

# Global publisher for dashboard conversation
conversation_pub = None

from states.move_robot import JointTrajectoryIterator, MoveRobotPart, MoveRobot
from states.action_planner import ConcurrentPlanAndVerify
from states.action_parser import ActionParser
from states.hand_control import HandControl, OpenHand, CloseHand, PreGraspHand
from states.grasp_refinement import GraspRefinement


def main():
    rospy.init_node("elmira_state_machine")

    rospy.loginfo("ELMiRA: Using MLLM v2 gateway")

    # Create a state machine
    sm = smach.StateMachine(outcomes=["succeeded", "aborted", "preempted"])

    # set topic names(TODO change into proper NICO paths TODO turn into rospy param?)
    MOTION_SUB_LEFT = "/left/open_manipulator_p/joint_states"
    MOTION_SRV_LEFT = "/left/open_manipulator_p/goal_joint_space_path"
    MOTION_SUB_RIGHT = "/right/open_manipulator_p/joint_states"
    MOTION_SRV_RIGHT = "/right/open_manipulator_p/goal_joint_space_path"
    MOTION_SUB_HEAD = "/NICOL/joint_states"
    MOTION_SRV_HEAD = "/NICOL/head/goal_joint_space_path"
    
    # Initialize conversation publisher (Latched so dashboard receives history on connect)
    global conversation_pub
    conversation_pub = rospy.Publisher("/elmira/conversation", String, queue_size=10, latch=True)

    # set initial userdata
    sm.userdata.system_message = ""
    sm.userdata.llm_input = ""
    # speech recognition
    sm.userdata.asr_detect_start = rospy.get_param("~detect_start", True)
    sm.userdata.asr_detect_stop = rospy.get_param("~detect_stop", True)
    sm.userdata.asr_start_timeout = rospy.get_param("~start_timeout", 0.0)
    sm.userdata.asr_min_duration = rospy.get_param("~min_duration", 3.0)
    sm.userdata.asr_max_duration = rospy.get_param("~max_duration", 30.0)
    sm.userdata.asr_min_period = rospy.get_param("~min_period", 3.0)
    sm.userdata.asr_live_text = rospy.get_param("~live_text", True)
    # robot motion
    sm.userdata.motion_init_pose = {
        "l_arm": {
            "names": [
                "l_shoulder_z",
                "l_shoulder_y",
                "l_arm_x",
                "l_elbow_y",
                "l_wrist_z",
                "l_wrist_x",
            ],
            "positions": [0.157, 0.0, 1.57, 1.57, 1.39, 0.0],
        },
        "r_arm": {
            "names": [
                "r_shoulder_z",
                "r_shoulder_y",
                "r_arm_x",
                "r_elbow_y",
                "r_wrist_z",
                "r_wrist_x",
            ],
            "positions": [-0.157, 0.0, -1.57, -1.57, -1.39, 0.0],
        },
        "head": {
            "names": [
                "head_z",
                "head_y",
            ],
            "positions": [0.0, 0.0],
        },
    }
    sm.userdata.motion_safe_names_left = [
        "l_shoulder_z", "l_shoulder_y", "l_arm_x",
        "l_elbow_y", "l_wrist_z", "l_wrist_x",
    ]
    sm.userdata.motion_safe_pose_left = [0.157, 0.0, 0.8203, 1.57, 1.39, 0.0]
    sm.userdata.motion_safe_names_right = [
        "r_shoulder_z", "r_shoulder_y", "r_arm_x",
        "r_elbow_y", "r_wrist_z", "r_wrist_x",
    ]
    sm.userdata.motion_safe_pose_right = [-0.157, 0.0, -0.8203, -1.57, -1.39, 0.0]
    sm.userdata.motion_look_down_names = ["head_z", "head_y"]
    # Note: Negative head_y looks DOWN, positive looks UP
    sm.userdata.motion_look_down_positions = [0.0, -0.5]
    base_table_z = 0.7
    offset_z = rospy.get_param("/elmira/offset_z", 0.0)
    sm.userdata.table_z = base_table_z + offset_z
    rospy.loginfo(f"Table Z set to {sm.userdata.table_z} (Base {base_table_z} + Offset {offset_z})")
    # TTS
    sm.userdata.tts_language = "en"
    sm.userdata.tts_pitch = 0.0
    sm.userdata.tts_speed = 1.0
    sm.userdata.tts_blocking = True

    # Add states
    with sm:
        # move to initial state
        @smach.cb_interface(output_keys=["llm_actions"], outcomes=["initial_pose"])
        def initial_pose_callback(userdata):
            userdata.llm_actions = [
                {"action": "initial_pose"},
                {"action": "speak", "text": "System initialized. I am listening."}
            ]
            if conversation_pub:
                conversation_pub.publish("ROBOT: System initialized. I am listening.")

            return "initial_pose"

        smach.StateMachine.add(
            "INIT",
            smach.CBState(initial_pose_callback),
            {"initial_pose": "LLM_RESPONSE_ITERATOR"},
        )

        # callback to post-process asr result
        def asr_result_callback(userdata, status, result):
            if status == GoalStatus.SUCCEEDED:
                if len(result.text) == 0:
                    rospy.logwarn("Empty speech result")
                    return "empty"
                else:
                    rospy.loginfo(f"USER: {result.text}")
                    if conversation_pub:
                        conversation_pub.publish(f"USER: {result.text}")
                    userdata.llm_input = f"USER: {result.text}"
                    return "succeeded"

        # listen for human command via speech asr ros action
        smach.StateMachine.add(
            "SPEECH_ASR",
            smach_ros.SimpleActionState(
                "speech_asr",
                PerformASRAction,
                goal_slots=[
                    "detect_start", "detect_stop", "start_timeout",
                    "min_duration", "max_duration", "min_period", "live_text",
                ],
                result_cb=asr_result_callback,
                output_keys=["llm_input"],
                outcomes=["empty"],
            ),
            transitions={"succeeded": "LLM_SPEECH_PROCESSOR", "empty": "SPEECH_ASR"},
            remapping={
                "detect_start": "asr_detect_start",
                "detect_stop": "asr_detect_stop",
                "start_timeout": "asr_start_timeout",
                "min_duration": "asr_min_duration",
                "max_duration": "asr_max_duration",
                "min_period": "asr_min_period",
                "live_text": "asr_live_text",
            },
        )

        # callback to post-process LLM response
        def llm_response_callback(userdata, response):
            rospy.loginfo(f"LLM output:\n{response.response}")
            actions_list = parse_llm_actions(response.response)
            userdata.llm_actions = actions_list

            # Publish robot response to conversation
            if conversation_pub and actions_list:
                for action in actions_list:
                    act_type = action.get("action")
                    if act_type == "speak":
                        text = action.get("text", "")
                        if text:
                            conversation_pub.publish(f"ROBOT: {text}")
                    elif act_type == "act":
                        obj = action.get("object", "object")
                        atype = action.get("type", "acting on")
                        conversation_pub.publish(f"ROBOT: *{atype.capitalize()} {obj}*")
                    elif act_type == "describe":
                        conversation_pub.publish("ROBOT: *Analyzing scene...*")
            return "succeeded"

        smach.StateMachine.add(
            "LLM_SPEECH_PROCESSOR",
            smach_ros.ServiceState(
                LLM_CHAT_SERVICE,
                PromptTextLLM,
                request_slots=["prompt"],
                response_cb=llm_response_callback,
                output_keys=["llm_actions"],
            ),
            remapping={"prompt": "llm_input"},
            transitions={
                "succeeded": "LLM_RESPONSE_ITERATOR",
            },
        )

        # iterate through llm response actions
        llm_response_it = smach.Iterator(
            outcomes=[
                "succeeded",
                "preempted",
                "aborted",
                "update_actions",
                "system_out",
                "quit",
            ],
            input_keys=[
                "motion_init_pose",
                "motion_look_down_names",
                "motion_look_down_positions",
                "llm_input",
                "llm_actions",
                "table_z",
                "tts_language",
                "tts_pitch",
                "tts_speed",
                "tts_blocking",
                "system_message",
            ],
            it=lambda: range(0, len(sm.userdata.llm_actions)),
            output_keys=["llm_actions", "system_message", "hand_action", "planning_group", "action_type", "real_x", "real_y", "target_object"],
            it_label="action_index",
            exhausted_outcome="succeeded",
        )
        with llm_response_it:
            execute_actions_sm = smach.StateMachine(
                outcomes=[
                    "succeeded",
                    "preempted",
                    "aborted",
                    "next_action",
                    "update_actions",
                    "system_out",
                    "quit",
                ],
                input_keys=[
                    "action_index",
                    "motion_init_pose",
                    "motion_look_down_names",
                    "motion_look_down_positions",
                    "llm_input",
                    "llm_actions",
                    "table_z",
                    "tts_language",
                    "tts_pitch",
                    "tts_speed",
                    "tts_blocking",
                    "system_message",
                ],
                output_keys=["llm_actions", "system_message", "hand_action", "planning_group", "action_type", "real_x", "real_y", "target_object"],
            )
            with execute_actions_sm:

                # parse next action
                smach.StateMachine.add(
                    "ACTION_PARSER",
                    ActionParser(),
                    transitions={
                        "speak": "TEXT_TO_SPEECH",
                        "act": "LOOK_DOWN_ACT",
                        "describe": "LOOK_DOWN_DESCRIBE",  # "LLM_SCENE_DESCRIPTION",
                        "quit": "quit",
                        "initial_pose": "JOINT_TRAJECTORY_ITERATOR",
                    },
                )

                # SPEAK ACTION
                smach.StateMachine.add(
                    "TEXT_TO_SPEECH",
                    smach_ros.ServiceState(
                        "nico/text_to_speech/say",
                        SayText,
                        request_slots=[
                            "text",
                            "language",
                            "pitch",
                            "speed",
                            "blocking",
                        ],
                        # response_slots=["duration"],
                    ),
                    remapping={
                        "text": "tts_text",
                        "language": "tts_language",
                        "pitch": "tts_pitch",
                        "speed": "tts_speed",
                        "blocking": "tts_blocking",
                    },
                    transitions={"succeeded": "next_action"},
                )

                # DESCRIBE ACTION
                smach.StateMachine.add(
                    "LOOK_DOWN_DESCRIBE",
                    MoveRobotPart(MOTION_SRV_HEAD, MOTION_SUB_HEAD),
                    remapping={
                        "names": "motion_look_down_names",
                        "positions": "motion_look_down_positions",
                    },
                    transitions={"succeeded": "LLM_SCENE_DESCRIPTION"},
                )

                def llm_scene_description_callback(userdata, response):
                    rospy.loginfo(f"LLM output:\n{response.response}")
                    actions_list = parse_llm_actions(response.response)
                    userdata.llm_actions = actions_list
                    
                    if conversation_pub:
                        for act in actions_list:
                            if act.get("action") == "speak":
                                conversation_pub.publish(f"ROBOT: {act.get('text', '')}")
                            elif act.get("action") == "describe":
                                conversation_pub.publish("ROBOT: *Describing scene...*")
                            elif act.get("action") == "act":
                                obj = act.get("object", "object")
                                conversation_pub.publish(f"ROBOT: *Acting on {obj}*")

                    return "succeeded"

                smach.StateMachine.add(
                    "LLM_SCENE_DESCRIPTION",
                    smach_ros.ServiceState(
                        LLM_VISION_SERVICE,
                        PromptVisionLLM,
                        response_cb=llm_scene_description_callback,
                        output_keys=["llm_actions"],
                    ),
                    transitions={
                        "succeeded": "update_actions",
                    },
                )

                # ACT ACTION
                smach.StateMachine.add(
                    "LOOK_DOWN_ACT",
                    MoveRobotPart(MOTION_SRV_HEAD, MOTION_SUB_HEAD),
                    remapping={
                        "names": "motion_look_down_names",
                        "positions": "motion_look_down_positions",
                    },
                    transitions={"succeeded": "PLAN_ACTION_TRAJECTORY"},
                )
                # plan action and verify if object is actually on the table
                smach.StateMachine.add(
                    "PLAN_ACTION_TRAJECTORY",
                    ConcurrentPlanAndVerify(),
                    {
                        "succeeded": "CHECK_PREGRASP",
                        "system_out": "system_out",
                    },
                )
                
                # Pre-grasp: open hand before grasp approach
                @smach.cb_interface(
                    input_keys=["action_type", "hand_action"],
                    outcomes=["needs_pregrasp", "no_pregrasp"],
                )
                def check_pregrasp_cb(userdata):
                    if userdata.action_type == "grasp":
                        rospy.loginfo("Pre-grasp: Opening hand before approach")
                        return "needs_pregrasp"
                    return "no_pregrasp"
                
                smach.StateMachine.add(
                    "CHECK_PREGRASP",
                    smach.CBState(check_pregrasp_cb),
                    {
                        "needs_pregrasp": "PREGRASP_OPEN_HAND",
                        "no_pregrasp": "JOINT_TRAJECTORY_ITERATOR",
                    },
                )
                
                smach.StateMachine.add(
                    "PREGRASP_OPEN_HAND",
                    OpenHand(),
                    {"succeeded": "JOINT_TRAJECTORY_ITERATOR", "failed": "JOINT_TRAJECTORY_ITERATOR"},
                )
                
                # execute movement (for grasp: only the approach pose)
                smach.StateMachine.add(
                    "JOINT_TRAJECTORY_ITERATOR",
                    JointTrajectoryIterator(
                        MOTION_SRV_HEAD,
                        MOTION_SUB_HEAD,
                        MOTION_SRV_LEFT,
                        MOTION_SUB_LEFT,
                        MOTION_SRV_RIGHT,
                        MOTION_SUB_RIGHT,
                    ),
                    {"succeeded": "CHECK_GRASP_REFINE"},
                )
                
                # Check if this was a grasp action that needs visual servoing refinement
                @smach.cb_interface(
                    input_keys=["action_type"],
                    outcomes=["needs_refinement", "done"],
                )
                def check_grasp_refine_cb(userdata):
                    action = str(userdata.action_type).lower()
                    if action in ["grasp", "grab", "pick", "take"]:
                        rospy.loginfo("Grasp detected: Starting multi-stage visual servoing refinement")
                        return "needs_refinement"
                    return "done"
                
                smach.StateMachine.add(
                    "CHECK_GRASP_REFINE",
                    smach.CBState(check_grasp_refine_cb),
                    {
                        "needs_refinement": "PREPARE_STAGE2",
                        "done": "next_action",
                    },
                )
                
                # Setup parameters for Stage 2 Macro-refinement (Right Eye, no close)
                @smach.cb_interface(
                    input_keys=[],
                    output_keys=["camera_eye", "close_hand"],
                    outcomes=["done"],
                )
                def prepare_stage2_cb(userdata):
                    userdata.camera_eye = "right"
                    userdata.close_hand = False
                    return "done"
                    
                smach.StateMachine.add(
                    "PREPARE_STAGE2",
                    smach.CBState(prepare_stage2_cb),
                    {"done": "GRASP_REFINEMENT_STAGE2"}
                )
                
                # Visual servoing Stage 2: right eye observation and horizontal slide
                smach.StateMachine.add(
                    "GRASP_REFINEMENT_STAGE2",
                    GraspRefinement(),
                    transitions={
                        "succeeded": "GRASP_STAGE2_EXECUTE",
                        "aborted": "next_action",  # Fall back on failure
                    },
                )
                
                # Execute Stage 2 Refinement
                smach.StateMachine.add(
                    "GRASP_STAGE2_EXECUTE",
                    JointTrajectoryIterator(
                        MOTION_SRV_HEAD, MOTION_SUB_HEAD,
                        MOTION_SRV_LEFT, MOTION_SUB_LEFT,
                        MOTION_SRV_RIGHT, MOTION_SUB_RIGHT,
                    ),
                    {"succeeded": "PREPARE_STAGE3"},
                )
                
                # Setup parameters for Stage 3 Micro-refinement (Left Eye, close hand)
                @smach.cb_interface(
                    input_keys=[],
                    output_keys=["camera_eye", "close_hand"],
                    outcomes=["done"],
                )
                def prepare_stage3_cb(userdata):
                    userdata.camera_eye = "left"
                    userdata.close_hand = True
                    return "done"
                    
                smach.StateMachine.add(
                    "PREPARE_STAGE3",
                    smach.CBState(prepare_stage3_cb),
                    {"done": "GRASP_REFINEMENT_STAGE3"}
                )
                
                # Visual servoing Stage 3: left eye unoccluded observation
                smach.StateMachine.add(
                    "GRASP_REFINEMENT_STAGE3",
                    GraspRefinement(),
                    transitions={
                        "succeeded": "GRASP_FINAL_EXECUTE",
                        "aborted": "VERIFY_GRASP",  # Try to grasp anyway if refinement fails
                    },
                )
                
                # Execute the final refined grasp trajectory (slide to close + lift)
                smach.StateMachine.add(
                    "GRASP_FINAL_EXECUTE",
                    JointTrajectoryIterator(
                        MOTION_SRV_HEAD,
                        MOTION_SUB_HEAD,
                        MOTION_SRV_LEFT,
                        MOTION_SUB_LEFT,
                        MOTION_SRV_RIGHT,
                        MOTION_SUB_RIGHT,
                    ),
                    {"succeeded": "VERIFY_GRASP"},
                )
                
                # Verify grasp success using palm sensor and override remaining LLM actions
                @smach.cb_interface(
                    input_keys=["planning_group", "llm_actions", "action_index", "target_object"],
                    output_keys=["llm_actions"],
                    outcomes=["next_action"],
                )
                def verify_grasp_cb(userdata):
                    from states.hand_control import PalmSensorMonitor
                    palm = PalmSensorMonitor()
                    rospy.sleep(0.5)  # Let sensor settle
                    
                    planning_group = userdata.planning_group or "r_arm"
                    side = "left" if "l_" in planning_group else "right"
                    grasped = palm.is_grasping(side)
                    reading = palm.get_reading(side)
                    
                    target = userdata.target_object
                    if isinstance(target, (list, tuple)):
                        target = target[0] if target else "the object"
                    
                    # Replace all remaining actions after the current grasp action
                    # with a truthful response based on sensor feedback
                    current_idx = userdata.action_index
                    kept_actions = userdata.llm_actions[:current_idx + 1]
                    
                    if grasped:
                        rospy.loginfo(f"VERIFY_GRASP: Success! Palm sensor={reading}, side={side}")
                        kept_actions.append({
                            "action": "speak",
                            "text": f"I've successfully grasped {target}."
                        })
                    else:
                        rospy.logwarn(f"VERIFY_GRASP: Failed! Palm sensor={reading}, side={side}")
                        kept_actions.append({
                            "action": "speak",
                            "text": f"I tried to grasp {target}, but I don't think I got it. "
                                    f"Could you move it closer and ask me to try again?"
                        })
                    
                    userdata.llm_actions = kept_actions
                    return "next_action"
                
                smach.StateMachine.add(
                    "VERIFY_GRASP",
                    smach.CBState(verify_grasp_cb),
                    {"next_action": "next_action"},
                )

            # close execute_actions_sm
            smach.Iterator.set_contained_state(
                "EXECUTE_ACTIONS", execute_actions_sm, loop_outcomes=["next_action"]
            )
        # close the llm_response_it
        smach.StateMachine.add(
            "LLM_RESPONSE_ITERATOR",
            llm_response_it,
            {
                "succeeded": "SPEECH_ASR",
                "aborted": "aborted",
                "update_actions": "LLM_RESPONSE_ITERATOR",
                "system_out": "LLM_SPEECH_PROCESSOR",
                "quit": "MOVE_TO_SHUTDOWN_POSITION",
            },
            remapping={"system_message": "llm_input"},
        )
        # move robot to safe position and shut down
        smach.StateMachine.add(
            "MOVE_TO_SHUTDOWN_POSITION",
            MoveRobot(
                MOTION_SRV_HEAD,
                MOTION_SUB_HEAD,
                MOTION_SRV_LEFT,
                MOTION_SUB_LEFT,
                MOTION_SRV_RIGHT,
                MOTION_SUB_RIGHT,
            ),
            {"movement_done": "DISABLE_TORQUE"},
            remapping={
                "names_head": "motion_look_down_names",
                "positions_head": "motion_look_down_positions",
                "names_left": "motion_safe_names_left",
                "positions_left": "motion_safe_pose_left",
                "names_right": "motion_safe_names_right",
                "positions_right": "motion_safe_pose_right",
            },
        )

        # disable torque
        @smach.cb_interface(outcomes=["quit"])
        def shutdown_callback(userdata):
            pub = rospy.Publisher("/nico/motion/disableTorqueAll", empty, queue_size=1)
            # wait until subscribers are connected to publish
            rate = rospy.Rate(10)
            while not rospy.is_shutdown():
                connections = pub.get_num_connections()
                if connections > 0:
                    pub.publish(empty())
                    break
                rate.sleep()
            return "quit"

        smach.StateMachine.add(
            "DISABLE_TORQUE",
            smach.CBState(shutdown_callback),
            {"quit": "succeeded"},
        )

    # Create and start the introspection server for visualization
    sis = smach_ros.IntrospectionServer("elmira_introspection", sm, "/ELMiRA")
    sis.start()
    # Execute state machine
    sm.execute()
    sis.stop()


if __name__ == "__main__":
    main()
