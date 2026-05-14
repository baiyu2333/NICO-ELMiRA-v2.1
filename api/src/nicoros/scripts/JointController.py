#!/usr/bin/env python3

import rospy
import math
from nicomsg.msg import sff
from open_manipulator_msgs.srv import SetJointPosition
from open_manipulator_msgs.msg import JointPosition


class JointController:

    def __init__(self, prefix, command_speed):
        self.pub = rospy.Publisher("/nico/motion/setAngle", sff, queue_size=10)
        self.command_speed = command_speed
        self.joint_srv = rospy.Service(
            prefix + "/goal_joint_space_path",
            SetJointPosition,
            self.set_joint_position,
        )
        rospy.loginfo(
            f"Started joint controller with prefix: {prefix}, command_speed={command_speed}"
        )

    def set_joint_position(self, request):
        joint_position = request.joint_position
        message = sff()
        for i, joint_name in enumerate(joint_position.joint_name):
            message.param1 = joint_name
            message.param2 = math.degrees(joint_position.position[i])
            message.param3 = self.command_speed
            rospy.loginfo(
                "JointController command: %s -> %.2f deg at speed %.3f",
                message.param1,
                message.param2,
                message.param3,
            )
            self.pub.publish(message)
        return True


def main():
    node = rospy.init_node("joint_controller_node", anonymous=True)
    command_speed = float(rospy.get_param("~command_speed", 0.03))
    command_speed = max(0.0, min(command_speed, 1.0))
    jc = JointController(rospy.get_param("~prefix", ""), command_speed)
    rospy.spin()


if __name__ == "__main__":
    main()
