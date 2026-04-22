#!/usr/bin/env python3

from os.path import dirname, abspath, join, pardir
import rospy
import torch

from coordinate_transfer_net import ImplicitCoordinateTransfer
from elmira.srv import CoordinateTransfer


class ImplicitCoordinateTransferServer:
    def __init__(
        self,
    ):
        rospy.init_node("implicit_transfer_server")
        # use GPU if possible
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rospy.loginfo(f"Using {device} device")
        # init implicit model
        self.implicit_model = ImplicitCoordinateTransfer(
            join(
                dirname(abspath(__file__)),
                pardir,
                "model_checkpoints/implicit_model_weights.pth",
            ),
            device,
        )
        # launch service
        rospy.Service(
            "image_to_real", CoordinateTransfer, self.image_to_real_request_handler
        )
        rospy.loginfo("ImplicitCoordinateTransfer started successfully")
        rospy.spin()

    def image_to_real_request_handler(self, request):
        rospy.loginfo(
            f"CoordTransfer: image_coords=({request.image_x:.4f}, {request.image_y:.4f})"
        )
        raw_real_x, raw_real_y = (
            self.implicit_model.derivative_free_optimizer(
                torch.unsqueeze(
                    torch.FloatTensor([request.image_x, request.image_y]), 0
                )
            )
            .squeeze()
            .cpu()
        )
        # Apply manual offsets for calibration
        offset_x = rospy.get_param("~offset_x", 0.0)
        offset_y = rospy.get_param("~offset_y", 0.0)
        
        real_x = raw_real_x + offset_x
        real_y = raw_real_y + offset_y
        
        rospy.loginfo(
            f"CoordTransfer: raw=({raw_real_x:.4f}, {raw_real_y:.4f}) "
            f"+ offset=({offset_x:.4f}, {offset_y:.4f}) "
            f"= final=({real_x:.4f}, {real_y:.4f})"
        )
        return float(real_x), float(real_y)


if __name__ == "__main__":
    ImplicitCoordinateTransferServer()
