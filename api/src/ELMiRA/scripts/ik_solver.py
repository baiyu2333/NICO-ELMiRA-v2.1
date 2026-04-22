#!/usr/bin/env python3
from evo_ik import EvoIK
import numpy as np
from elmira.srv import InverseKinematics, InverseKinematicsResponse
from elmira.msg import JointPosition
from os.path import dirname, abspath, join, pardir
import rospy
import torch


# Monkey patch to fix evotorch error with bounded problems
try:
    import evotorch.core
    def dummy_ensure_unbounded(self):
        pass
    evotorch.core.Problem.ensure_unbounded = dummy_ensure_unbounded
    rospy.loginfo("Applied monkey patch to evotorch.core.Problem.ensure_unbounded")
except ImportError:
    rospy.logwarn("Could not import evotorch to patch ensure_unbounded")

class KinematicsServer:
    def __init__(self):
        rospy.init_node("kinematics_server")
        urdf_dir = join(dirname(abspath(__file__)), pardir, "urdf")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            rospy.loginfo("EvoIK using CUDA")
        else:
            self.device = torch.device("cpu")
            rospy.logwarn("EvoIK using CPU (CUDA not available)")
        self.left_arm = EvoIK(
            join(urdf_dir, "nico_left_arm.urdf"), "left_tcp", device=self.device
        )
        self.right_arm = EvoIK(
            join(urdf_dir, "nico_right_arm.urdf"), "right_tcp", device=self.device
        )
        rospy.Service(
            "inverse_kinematics", InverseKinematics, self.get_inverse_kinematics
        )
        rospy.loginfo("EvoIK started successfully")
        rospy.spin()

    def get_inverse_kinematics(self, request):
        if request.planning_group == "l_arm":
            solver = self.left_arm
        elif request.planning_group == "r_arm":
            solver = self.right_arm
        else:
            rospy.logerr(f"Unknown planning group {request.planning_group}")
            return
        # ensure that initial position is in the right order
        joint_ids = np.argsort(request.initial_position.joint_name)
        initial_joints = torch.tensor(request.initial_position.position)[
            joint_ids[
                np.searchsorted(
                    request.initial_position.joint_name,
                    solver.joint_names[:6],
                    sorter=joint_ids,
                )
            ]
        ]
        # solve trajectory
        results = []
        for i, pose in enumerate(request.poses):
            ik_result = JointPosition()
            ik_result.joint_name = solver.joint_names[:6]
            pos = pose.position
            quat = pose.orientation
            target_pos = torch.tensor([pos.x, pos.y, pos.z]).to(self.device)
            target_quat = torch.tensor([quat.w, quat.x, quat.y, quat.z]).to(self.device)
            try:
                ik_result.position = solver.inverse_kinematics(
                    target_pos,
                    target_quat,
                    initial_joints=initial_joints,
                    max_steps=100,
                )
            except Exception as e:
                rospy.logwarn(f"IK solver numeric crash (likely Cholesky fail at limit): {e}. Falling back to last known safe pose.")
                ik_result.position = initial_joints
            # Check IK solution quality
            dist_err, angle_err = solver.get_euclidean_errors(
                ik_result.position, target_pos, target_quat
            )
            d = dist_err.detach().cpu().item()
            a = torch.rad2deg(angle_err).detach().cpu().item()
            if d > 0.01 or a > 20.0:
                rospy.logwarn(
                    f"IK solution POOR for pose {i} "
                    f"(target=({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})): "
                    f"position_error={d:.4f}m, orientation_error={a:.1f}°. "
                    f"Target may be outside reachable workspace!"
                )
            else:
                rospy.loginfo(
                    f"IK solution OK for pose {i}: "
                    f"pos_err={d:.4f}m, orient_err={a:.1f}°"
                )
            results.append(ik_result)
            # use solution as starting point for the next one
            initial_joints = ik_result.position
        return InverseKinematicsResponse(results)

    # def get_forward_kinematic(self, solver):
    #     rad_angles = torch.tensor(
    #         self.get_motor_angles(solver.joint_names[:6])
    #     ).deg2rad()
    #     return solver.forward_kinematics(rad_angles)


if __name__ == "__main__":
    KinematicsServer()
