#!/usr/bin/env python3

import argparse
import logging
import math
import struct
import sys
import threading
import time
from os.path import abspath, dirname, join

import nicomsg.msg
import nicomsg.srv
import rospy
import sensor_msgs.msg
from nicomotion.Motion import Motion
from RosLoggingHandler import RosLoggingHandler
from std_srvs.srv import Empty

try:
    from nicomoveit import moveitWrapper
except ImportError:
    pass


class NicoRosMotion:
    """
    The NicoRosMotion class exposes the functions of :class:`nicomotion.Motion`
    to ROS and
    periodically publishes the current joint states
    """

    @staticmethod
    def getConfig():
        """
        Returns a default config dict

        :return: dict
        """
        return {
            "robotMotorFile": join(
                dirname(abspath(__file__)),
                "../../../..",
                "json/nico_humanoid_upper.json",
            ),
            "vrep": False,
            "vrepHost": "127.0.0.1",
            "vrepPort": 19997,
            "vrepScene": "",
            "rostopicName": "/nico/motion",
            "jointStateName": "/joint_states",
            "fakeExecution": False,
            "usePyrep": False,
            "pyrep": False,
            "headless": False,
            "disabledMotorIds": [24, 26, 28, 30, 32],
        }

    def __init__(self, config=None):
        """
        RosNicoMotion provides :class:`nicomotion.Motion` functions over ROS
        and periodically publishes the current joint states

        :param config: Configuration of the :class:`nicomotion.Motion` and
                       RosNicoMotion interface
        :type config: dict
        """
        self.logger = logging.getLogger(__name__)
        self._running = False
        self.robot = None
        if config is None:
            config = NicoRosMotion.getConfig()

        if rospy.has_param(config["rostopicName"] + "/robotMotorFile"):
            config["robotMotorFile"] = rospy.get_param(
                config["rostopicName"] + "/robotMotorFile"
            )
        if rospy.has_param(config["rostopicName"] + "/vrep"):
            config["vrep"] = rospy.get_param(config["rostopicName"] + "/vrep")
        if rospy.has_param(config["rostopicName"] + "/vrepScene"):
            config["vrepScene"] = rospy.get_param(config["rostopicName"] + "/vrepScene")
        if rospy.has_param(config["rostopicName"] + "/fakeExecution"):
            config["fakeExecution"] = rospy.get_param(
                config["rostopicName"] + "/fakeExecution"
            )
        if rospy.has_param(config["rostopicName"] + "/pyrep"):
            config["pyrep"] = rospy.get_param(config["rostopicName"] + "/pyrep")
        if rospy.has_param(config["rostopicName"] + "/headless"):
            config["headless"] = rospy.get_param(config["rostopicName"] + "/headless")
        if rospy.has_param(config["rostopicName"] + "/disabledMotorIds"):
            config["disabledMotorIds"] = rospy.get_param(
                config["rostopicName"] + "/disabledMotorIds"
            )

        # init Motion
        self.logger.info("-- Init NicoRosMotion --")
        if config["pyrep"]:
            vrepConfig = Motion.pyrepConfig()
            vrepConfig["vrep_scene"] = config["vrepScene"]
            vrepConfig["headless"] = config["headless"]
        else:
            vrepConfig = Motion.vrepRemoteConfig()
            vrepConfig["vrep_scene"] = config["vrepScene"]
            vrepConfig["vrep_host"] = config["vrepHost"]
            vrepConfig["vrep_port"] = config["vrepPort"]
        disabled_ids = [int(x) for x in config.get("disabledMotorIds", [])]
        self.robot = Motion(
            motorConfig=config["robotMotorFile"],
            vrep=config["vrep"],
            vrepConfig=vrepConfig,
            ignoreMissing=True,
            disabled_ids=disabled_ids,
        )

        # init ROS
        self.logger.debug("Init ROS")
        rospy.init_node("nicorosmotion", anonymous=True)

        # setup subscriber
        self.logger.debug("Init subscriber")
        rospy.Subscriber(
            "%s/openHand" % config["rostopicName"], nicomsg.msg.s, self._ROSPY_openHand
        )
        rospy.Subscriber(
            "%s/closeHand" % config["rostopicName"],
            nicomsg.msg.s,
            self._ROSPY_closeHand,
        )
        rospy.Subscriber(
            "%s/enableForceControlAll" % config["rostopicName"],
            nicomsg.msg.i,
            self._ROSPY_enableForceControlAll,
        )
        rospy.Subscriber(
            "%s/disableForceControlAll" % config["rostopicName"],
            nicomsg.msg.empty,
            self._ROSPY_disableForceControlAll,
        )
        rospy.Subscriber(
            "%s/enableForceControl" % config["rostopicName"],
            nicomsg.msg.si,
            self._ROSPY_enableForceControl,
        )
        rospy.Subscriber(
            "%s/disableForceControl" % config["rostopicName"],
            nicomsg.msg.s,
            self._ROSPY_disableForceControl,
        )
        rospy.Subscriber(
            "%s/setAngle" % config["rostopicName"],
            nicomsg.msg.sff,
            self._ROSPY_setAngle,
        )
        rospy.Subscriber(
            "%s/changeAngle" % config["rostopicName"],
            nicomsg.msg.sff,
            self._ROSPY_changeAngle,
        )
        rospy.Subscriber(
            "%s/setMaximumSpeed" % config["rostopicName"],
            nicomsg.msg.f,
            self._ROSPY_setMaximumSpeed,
        )
        rospy.Subscriber(
            "%s/setStiffness" % config["rostopicName"],
            nicomsg.msg.sf,
            self._ROSPY_setStiffness,
        )
        rospy.Subscriber(
            "%s/setPID" % config["rostopicName"], nicomsg.msg.sfff, self._ROSPY_setPID
        )
        rospy.Subscriber(
            "%s/enableTorque" % config["rostopicName"],
            nicomsg.msg.s,
            self._ROSPY__enableTorque,
        )
        rospy.Subscriber(
            "%s/disableTorque" % config["rostopicName"],
            nicomsg.msg.s,
            self._ROSPY__disableTorque,
        )
        rospy.Subscriber(
            "%s/enableTorqueAll" % config["rostopicName"],
            nicomsg.msg.empty,
            self._ROSPY__enableTorqueAll,
        )
        rospy.Subscriber(
            "%s/disableTorqueAll" % config["rostopicName"],
            nicomsg.msg.empty,
            self._ROSPY__disableTorqueAll,
        )
        rospy.Subscriber(
            "%s/toSafePosition" % config["rostopicName"],
            nicomsg.msg.empty,
            self._ROSPY__toSafePosition,
        )
        # XL-320 Protocol 2.0 left hand control
        # Uses nicomsg.msg.sff: param1=motor_id (as string), param2=register, param3=value
        rospy.Subscriber(
            "%s/xl320_cmd" % config["rostopicName"],
            nicomsg.msg.sff,
            self._ROSPY_xl320_cmd,
        )
        self._xl320_lock = threading.Lock()

        # setup services
        self.logger.debug("Init services")
        rospy.Service(
            "%s/getConfig" % config["rostopicName"],
            nicomsg.srv.GetString,
            self._ROSPY_getConfig,
        )
        rospy.Service(
            "%s/getVrep" % config["rostopicName"],
            nicomsg.srv.GetString,
            self._ROSPY_getVrep,
        )
        rospy.Service(
            "%s/getAngle" % config["rostopicName"],
            nicomsg.srv.GetValue,
            self._ROSPY_getAngle,
        )
        rospy.Service(
            "%s/getPose" % config["rostopicName"],
            nicomsg.srv.GetValues,
            self._ROSPY_getPose,
        )
        rospy.Service(
            "%s/getJointNames" % config["rostopicName"],
            nicomsg.srv.GetNames,
            self._ROSPY_getJointNames,
        )
        rospy.Service(
            "%s/getAngleUpperLimit" % config["rostopicName"],
            nicomsg.srv.GetValue,
            self._ROSPY_getAngleUpperLimit,
        )
        rospy.Service(
            "%s/getAngleLowerLimit" % config["rostopicName"],
            nicomsg.srv.GetValue,
            self._ROSPY_getAngleLowerLimit,
        )
        rospy.Service(
            "%s/getTorqueLimit" % config["rostopicName"],
            nicomsg.srv.GetValue,
            self._ROSPY_getTorqueLimit,
        )
        rospy.Service(
            "%s/getTemperature" % config["rostopicName"],
            nicomsg.srv.GetValue,
            self._ROSPY_getTemperature,
        )
        rospy.Service(
            "%s/getCurrent" % config["rostopicName"],
            nicomsg.srv.GetValue,
            self._ROSPY_getCurrent,
        )
        rospy.Service(
            "%s/getStiffness" % config["rostopicName"],
            nicomsg.srv.GetValue,
            self._ROSPY_getStiffness,
        )
        rospy.Service(
            "%s/getPID" % config["rostopicName"], nicomsg.srv.GetPID, self._ROSPY_getPID
        )

        rospy.Service(
            "%s/nextSimulationStep" % config["rostopicName"],
            Empty,
            self._ROSPY__nextSimulationStep,
        )
        rospy.Service(
            "%s/startSimulation" % config["rostopicName"],
            Empty,
            self._ROSPY__startSimulation,
        )
        rospy.Service(
            "%s/stopSimulation" % config["rostopicName"],
            Empty,
            self._ROSPY__stopSimulation,
        )

        # setup class variables
        self._running = True
        self.jsonConfig = self.robot.getConfig()
        self.vrep = self.robot.getVrep()
        self.config = config
        self.fakeJointStates = {}

        # setup publishers
        self.logger.debug("Init publishers")
        self._palm_publisher_left = rospy.Publisher(
            "%s/palm_sensor/left" % config["rostopicName"],
            nicomsg.msg.i,
            queue_size=10,
        )
        self._palm_publisher_right = rospy.Publisher(
            "%s/palm_sensor/right" % config["rostopicName"],
            nicomsg.msg.i,
            queue_size=10,
        )
        self._palm_thread = threading.Thread(target=self._palm_sensor_publisher)
        self._palm_thread.start()

        self._publisher_right = rospy.Publisher(
            "/right/open_manipulator_p/joint_states",
            sensor_msgs.msg.JointState,
            queue_size=10,
        )
        self._publisher_left = rospy.Publisher(
            "/left/open_manipulator_p/joint_states",
            sensor_msgs.msg.JointState,
            queue_size=10,
        )
        self._publisher_head = rospy.Publisher(
            "/NICOL/joint_states",
            sensor_msgs.msg.JointState,
            queue_size=10,
        )
        self._joint_state_thread = threading.Thread(target=self._joint_state_publisher)
        self._joint_state_thread.start()

        if "nicomoveit.moveitWrapper" in sys.modules:
            self._jointStatePublisher = rospy.Publisher(
                "%s" % config["jointStateName"],
                sensor_msgs.msg.JointState,
                queue_size=1,
            )
            # a asynchronous thread is seperated from the main thread
            self._jointStateThread = threading.Thread(target=self._sendJointState)
            self._jointStateThread.start()
        else:
            self._jointStateThread = None

        # wait for messages
        self.logger.info("-- All done --")
        self.logger.info("XL-320 left hand control available on %s/xl320_cmd" % config["rostopicName"])

    def stop(self):
        self._running = False
        if self._jointStateThread:
            self._jointStateThread.join()
        self._palm_thread.join()

    def _ROSPY_openHand(self, message):
        """
        Callback handle for :meth:`nicomotion.Motion.openHand`

        :param message: ROS message
        :type message: nicomsg.msg.s
        """
        self.robot.openHand(message.param1)

    def _ROSPY_closeHand(self, message):
        """
        Callback handle for :meth:`nicomotion.Motion.closeHand`

        :param message: ROS message
        :type message: nicomsg.msg.s
        """
        self.robot.closeHand(message.param1)

    def _ROSPY_enableForceControlAll(self, message):
        """
        Callback handle for :meth:`nicomotion.Motion.enableForceControlAll`

        :param message: ROS message
        :type message: nicomsg.msg.i
        """
        self.robot.enableForceControlAll(message.param1)

    def _ROSPY_disableForceControlAll(self, message):
        """
        Callback handle for :meth:`nicomotion.Motion.disableForceControlAll`

        :param message: ROS message
        :type message: nicomsg.msg.empty
        """
        self.robot.disableForceControlAll()

    def _ROSPY_enableForceControl(self, message):
        """
        Callback handle for :meth:`nicomotion.Motion.enableForceControl`

        :param message: ROS message
        :type message: nicomsg.msg.si
        """
        self.robot.enableForceControl(message.param1, message.param2)

    def _ROSPY_disableForceControl(self, message):
        """
        Callback handle for :meth:`nicomotion.Motion.disableForceControl`

        :param message: ROS message
        :type message: nicomsg.msg.s
        """
        self.robot.disableForceControl(message.param1)

    def _ROSPY_setAngle(self, message):
        """
        Callback handle for :meth:`nicomotion.Motion.setAngles`

        :param message: ROS message
        :type message: nicomsg.msg.sff
        """
        self.fakeJointStates[message.param1] = message.param2
        self.robot.setAngle(message.param1, message.param2, message.param3)

    def _ROSPY_changeAngle(self, message):
        """
        Callback handle for :meth:`nicomotion.Motion.changeAngles`

        :param message: ROS message
        :type message: nicomsg.msg.sff
        """
        self.robot.changeAngle(message.param1, message.param2, message.param3)

    def _ROSPY_getVrep(self, message):
        """
        Callback handle for :meth:`nicomotion.Motion.getVrep`

        :param message: ROS message
        :type message: nicomsg.srv.GetString
        :return: 'True' if vrep is used, 'False' if real NICO is used
        :rtype: string
        """
        return str(self.robot.getVrep())

    def _ROSPY_getConfig(self, message):
        """
        Callback handle for :meth:`nicomotion.Motion.getConfig`

        :param message: ROS message
        :type message: nicomsg.srv.GetString
        :return: Dictionary with motor config that has been converted to a
                 string
        :rtype: string
        """
        return str(self.robot.getConfig())

    def _ROSPY_getAngle(self, message):
        """
        Callback handle for :meth:`nicomotion.Motion.getAngle`

        :param message: ROS message
        :type message: nicomsg.srv.GetValue
        :return: Angle of requested joint
        :rtype: float
        """
        return self.robot.getAngle(message.param1)

    def _ROSPY_getPose(self, message):
        """
        Callback handle for :meth:`nicomotion.Motion.getPose`

        :param message: ROS message
        :type message: nicomsg.srv.GetValues
        :return: Position of the requestet object in x,y,z coordinates
        :rtype: list
        """
        return [self.robot.getPose(message.param1)]

    def _ROSPY_getJointNames(self, message):
        """
        Callback handle for :meth:`nicomotion.Motion.getJointNames`

        :param message: ROS message
        :type message: nicomsg.srv.GetNames
        :return: List of joint names
        :rtype: list
        """
        return [self.robot.getJointNames()]

    def _ROSPY_getAngleUpperLimit(self, message):
        """
        Callback handle for :meth:`nicomotion.Motion.getAngleUpperLimit`

        :param message: ROS message
        :type message: nicomsg.srv.GetValue
        :return: Angle upper limit of requested joint
        :rtype: float
        """
        return self.robot.getAngleUpperLimit(message.param1)

    def _ROSPY_getAngleLowerLimit(self, message):
        """
        Callback handle for :meth:`nicomotion.Motion.getAngleLowerLimit`

        :param message: ROS message
        :type message: nicomsg.srv.GetValue
        :return: Angle lower limit of requested joint
        :rtype: float
        """
        return self.robot.getAngle(message.param1)

    def _ROSPY_getTorqueLimit(self, message):
        """
        Callback handle for :meth:`nicomotion.Motion.getTorqueLimit`

        :param message: ROS message
        :type message: nicomsg.srv.GetValue
        :return: Torque limit of requested joint
        :rtype: float
        """
        return self.robot.getTorqueLimit(message.param1)

    def _ROSPY_getTemperature(self, message):
        """
        Callback handle for :meth:`nicomotion.Motion.getTemperature`

        :param message: ROS message
        :type message: nicomsg.srv.GetValue
        :return: Temperature of requested joint
        :rtype: float
        """
        return self.robot.getTemperature(message.param1)

    def _ROSPY_getCurrent(self, message):
        """
        Callback handle for :meth:`nicomotion.Motion.getCurrent`

        :param message: ROS message
        :type message: nicomsg.srv.GetValue
        :return: Current of requested joint
        :rtype: float
        """
        return self.robot.getCurrent(message.param1)

    def _ROSPY_setMaximumSpeed(self, message):
        """
        Callback handle for :meth:`nicomotion.Motion.setMaximumSpeed`

        :param message: ROS message
        :type message: nicomsg.msg.f
        """
        self.robot.setMaximumSpeed(message.param1)

    def _ROSPY_setStiffness(self, message):
        """
        Callback handle for :meth:`nicomotion.Motion.setStiffness`

        :param message: ROS message
        :type message: nicomsg.msg.sf
        """
        self.robot.setStiffness(message.param1, message.param2)

    def _ROSPY_getStiffness(self, message):
        """
        Callback handle for :meth:`nicomotion.Motion.getStiffness`

        :param message: ROS message
        :type message: nicomsg.srv.GetValue
        :return: Stiffness of requested joint
        :rtype: float
        """
        return self.robot.getStiffness(message.param1)

    def _ROSPY_setPID(self, message):
        """
        Callback handle for :meth:`nicomotion.Motion.setPID`

        :param message: ROS message
        :type message: nicomsg.msg.sfff
        """
        self.robot.setPID(
            message.param1, message.param2, message.param3, message.param4
        )

    def _ROSPY_getPID(self, message):
        """
        Callback handle for :meth:`nicomotion.Motion.getPID`

        :param message: ROS message
        :type message: nicomsg.srv.GetPID
        :return: Tuple: (p, i, d)
        :rtype: tuple
        """
        return self.robot.getPID(message.param1)

    def _ROSPY__enableTorque(self, message):
        """
        Callback handle for :meth:`nicomotion.Motion.enableTorque`

        :param message: ROS message
        :type message: nicomsg.msg.s
        """
        self.robot.enableTorque(message.param1)

    def _ROSPY__disableTorque(self, message):
        """
        Callback handle for :meth:`nicomotion.Motion.disableTorque`

        :param message: ROS message
        :type message: nicomsg.msg.s
        """
        self.robot.disableTorque(message.param1)

    def _ROSPY__enableTorqueAll(self, message):
        """
        Callback handle for :meth:`nicomotion.Motion.enableTorqueAll`

        :param message: ROS message
        :type message: nicomsg.msg.empty
        """
        self.robot.enableTorqueAll()

    def _ROSPY__disableTorqueAll(self, message):
        """
        Callback handle for :meth:`nicomotion.Motion.disableTorqueAll`

        :param message: ROS message
        :type message: nicomsg.msg.empty
        """
        self.robot.disableTorqueAll()

    def _ROSPY__toSafePosition(self, message):
        """
        Callback handle for :meth:`nicomotion.Motion.toSafePosition`

        :param message: ROS message
        :type message: nicomsg.msg.empty
        """
        self.robot.toSafePosition()

    def _ROSPY__nextSimulationStep(self, message):
        """
        Callback handle for :meth:`nicomotion.Motion.nextSimulationStep`

        :param message: ROS message
        :type message: std_srvs.srv.Empty
        """
        self.robot.nextSimulationStep()
        return []

    def _ROSPY__startSimulation(self, message):
        """
        Callback handle for :meth:`nicomotion.Motion.startSimulation`

        :param message: ROS message
        :type message: std_srvs.srv.Empty
        """
        self.robot.startSimulation()
        return []

    def _ROSPY__stopSimulation(self, message):
        """
        Callback handle for :meth:`nicomotion.Motion.stopSimulation`

        :param message: ROS message
        :type message: std_srvs.srv.Empty
        """
        self.robot.stopSimulation()
        return []

    def _ROSPY_xl320_cmd(self, message):
        """Send a raw XL-320 Protocol 2.0 write command.
        
        Uses the existing pypot serial connection to avoid port contention.
        message.param1 = motor_id (as string, e.g. '33')
        message.param2 = register address (float, e.g. 24.0 for torque enable)
        message.param3 = value (float, e.g. 1.0)
        """
        motor_id = int(message.param1)
        register = int(message.param2)
        value = int(message.param3)
        
        # Determine data size from register
        # Register 24 = Torque Enable (1 byte)
        # Register 30 = Goal Position (2 bytes)
        # Register 32 = Moving Speed (2 bytes) 
        if register == 24:
            data = struct.pack('B', value & 0xFF)
        else:
            data = struct.pack('<H', value & 0xFFFF)
        
        # Build Protocol 2.0 packet
        packet = self._build_xl320_packet(motor_id, register, data)
        
        # Send through pypot's serial connection with proper locking
        with self._xl320_lock:
            try:
                # Access the underlying serial port from pypot's robot
                serial_port = None
                for controller in self.robot._robot._controllers:
                    if hasattr(controller, 'io') and hasattr(controller.io, '_serial'):
                        serial_port = controller.io._serial
                        break
                
                if serial_port is None:
                    self.logger.error("XL320: Could not find serial port from pypot")
                    return
                
                # Write directly to the serial port
                serial_port.write(packet)
                serial_port.flush()
                # Do NOT read. Let pypot's read thread discard the Protocol 2.0 response.
                
            except Exception as e:
                self.logger.error(f"XL320 cmd error (id={motor_id}, reg={register}, val={value}): {e}")
    
    @staticmethod
    def _build_xl320_packet(motor_id, register_addr, data_bytes):
        """Build a Dynamixel Protocol 2.0 Write instruction packet."""
        instruction = 0x03  # Write
        params = struct.pack('<H', register_addr) + data_bytes
        length = len(params) + 3  # instruction + params + 2 CRC bytes
        
        packet = bytearray([0xFF, 0xFF, 0xFD, 0x00, motor_id])
        packet += struct.pack('<H', length)
        packet.append(instruction)
        packet += params
        
        crc = NicoRosMotion._crc16_xl320(packet)
        packet += struct.pack('<H', crc)
        return bytes(packet)
    
    @staticmethod
    def _crc16_xl320(data):
        """CRC-16 for Dynamixel Protocol 2.0."""
        t = [
            0x0000,0x8005,0x800F,0x000A,0x801B,0x001E,0x0014,0x8011,
            0x8033,0x0036,0x003C,0x8039,0x0028,0x802D,0x8027,0x0022,
            0x8063,0x0066,0x006C,0x8069,0x0078,0x807D,0x8077,0x0072,
            0x0050,0x8055,0x805F,0x005A,0x804B,0x004E,0x0044,0x8041,
            0x80C3,0x00C6,0x00CC,0x80C9,0x00D8,0x80DD,0x80D7,0x00D2,
            0x00F0,0x80F5,0x80FF,0x00FA,0x80EB,0x00EE,0x00E4,0x80E1,
            0x00A0,0x80A5,0x80AF,0x00AA,0x80BB,0x00BE,0x00B4,0x80B1,
            0x8093,0x0096,0x009C,0x8099,0x0088,0x808D,0x8087,0x0082,
            0x8183,0x0186,0x018C,0x8189,0x0198,0x819D,0x8197,0x0192,
            0x01B0,0x81B5,0x81BF,0x01BA,0x81AB,0x01AE,0x01A4,0x81A1,
            0x01E0,0x81E5,0x81EF,0x01EA,0x81FB,0x01FE,0x01F4,0x81F1,
            0x81D3,0x01D6,0x01DC,0x81D9,0x01C8,0x81CD,0x81C7,0x01C2,
            0x0140,0x8145,0x814F,0x014A,0x815B,0x015E,0x0154,0x8151,
            0x8173,0x0176,0x017C,0x8179,0x0168,0x816D,0x8167,0x0162,
            0x8123,0x0126,0x012C,0x8129,0x0138,0x813D,0x8137,0x0132,
            0x0110,0x8115,0x811F,0x011A,0x810B,0x010E,0x0104,0x8101,
            0x8303,0x0306,0x030C,0x8309,0x0318,0x831D,0x8317,0x0312,
            0x0330,0x8335,0x833F,0x033A,0x832B,0x032E,0x0324,0x8321,
            0x0360,0x8365,0x836F,0x036A,0x837B,0x037E,0x0374,0x8371,
            0x8353,0x0356,0x035C,0x8359,0x0348,0x834D,0x8347,0x0342,
            0x03C0,0x83C5,0x83CF,0x03CA,0x83DB,0x03DE,0x03D4,0x83D1,
            0x83F3,0x03F6,0x03FC,0x83F9,0x03E8,0x83ED,0x83E7,0x03E2,
            0x83A3,0x03A6,0x03AC,0x83A9,0x03B8,0x83BD,0x83B7,0x03B2,
            0x0390,0x8395,0x839F,0x039A,0x838B,0x038E,0x0384,0x8381,
            0x0280,0x8285,0x828F,0x028A,0x829B,0x029E,0x0294,0x8291,
            0x82B3,0x02B6,0x02BC,0x82B9,0x02A8,0x82AD,0x82A7,0x02A2,
            0x82E3,0x02E6,0x02EC,0x82E9,0x02F8,0x82FD,0x82F7,0x02F2,
            0x02D0,0x82D5,0x82DF,0x02DA,0x82CB,0x02CE,0x02C4,0x82C1,
            0x8243,0x0246,0x024C,0x8249,0x0258,0x825D,0x8257,0x0252,
            0x0270,0x8275,0x827F,0x027A,0x826B,0x026E,0x0264,0x8261,
            0x0220,0x8225,0x822F,0x022A,0x823B,0x023E,0x0234,0x8231,
            0x8213,0x0216,0x021C,0x8219,0x0208,0x820D,0x8207,0x0202,
        ]
        c = 0
        for b in data:
            c = (c << 8) ^ t[((c >> 8) ^ b) & 0xFF]
            c &= 0xFFFF
        return c

    def _joint_state_publisher(self):
        r = rospy.Rate(50)  # 50hz
        pubs = self._publisher_head, self._publisher_left, self._publisher_right
        chains = (
            ["head_z", "head_y"],
            [
                "l_shoulder_z",
                "l_shoulder_y",
                "l_arm_x",
                "l_elbow_y",
                "l_wrist_z",
                "l_wrist_x",
                "l_indexfingers_x",
                "l_thumb_x",
            ],
            [
                "r_shoulder_z",
                "r_shoulder_y",
                "r_arm_x",
                "r_elbow_y",
                "r_wrist_z",
                "r_wrist_x",
                "r_indexfingers_x",
                "r_thumb_x",
            ],
        )
        while not rospy.is_shutdown():
            for i, pub in enumerate(pubs):
                message = sensor_msgs.msg.JointState()
                message.name = chains[i]
                message.position = [
                    math.radians(self.robot.getAngle(j)) for j in chains[i]
                ]
                message.velocity = [self.robot.getSpeed(j) for j in chains[i]]
                message.header.stamp = rospy.get_rostime()
                pub.publish(message)
            r.sleep()

    def _sendJointState(self):
        """
        Loop for sending the current joint state
        """
        while self._running:
            if rospy.has_param(self.config["rostopicName"] + "/fakeExecution"):
                self.config["fakeExecution"] = rospy.get_param(
                    self.config["rostopicName"] + "/fakeExecution"
                )
            message = sensor_msgs.msg.JointState()
            message.name = []
            message.position = []
            message.effort = []
            joints = self.robot.getJointNames()

            for joint in joints:
                if self.config["fakeExecution"] and joint in self.fakeJointStates:
                    value = self.fakeJointStates[joint]
                else:
                    value = self.robot.getAngle(joint)
                message.name += [joint]
                value = moveitWrapper.nicoToRosAngle(
                    joint, value, self.jsonConfig, self.vrep
                )
                rospy.loginfo(joint + " " + str(value))
                message.position += [value]
                # message.effort += [self.robot.getLoad(joint)]

            # to avoid a warning about missing joint values a joint value of
            # 0.0 is published for the joints with unknown joint value
            joints_without_motor = [
                "l_indexfinger_1st_x",
                "l_indexfinger_2nd_x",
                "l_ringfingers_x",
                "l_ringfinger_1st_x",
                "l_ringfinger_2nd_x",
                "l_thumb_1st_x",
                "l_thumb_2nd_x",
                "r_indexfinger_1st_x",
                "r_indexfinger_2nd_x",
                "r_ringfingers_x",
                "r_ringfinger_1st_x",
                "r_ringfinger_2nd_x",
                "r_thumb_1st_x",
                "r_thumb_2nd_x",
            ]
            for joint in joints_without_motor:
                message.name += [joint]
                value = 0.0
                message.position += [value]

            message.header.stamp = rospy.get_rostime()
            self._jointStatePublisher.publish(message)
            time.sleep(0.25)

    def __del__(self):
        if self._running:
            self.stop()
        del self.robot

    def _palm_sensor_publisher(self):
        r = rospy.Rate(10)  # 10hz
        while not rospy.is_shutdown():
            left = self.robot.getPalmSensorReading("l")
            right = self.robot.getPalmSensorReading("r")
            self._palm_publisher_left.publish(left)
            self._palm_publisher_right.publish(right)
            r.sleep()


if __name__ == "__main__":
    config = NicoRosMotion.getConfig()

    # Parse command line
    parser = argparse.ArgumentParser(description="NICO ROS motion interface")
    parser.add_argument(
        "--log-level",
        dest="logLevel",
        help="Sets log level. Default: INFO",
        type=str,
        default="INFO",
    )
    parser.add_argument(
        "-m",
        "--motor-file",
        dest="robotMotorFile",
        help=("Path to robot motor file. Default: " + config["robotMotorFile"]),
        type=str,
    )
    parser.add_argument(
        "-v",
        "--vrep",
        dest="vrep",
        help="Connect to VREP rather than to a real robot",
        action="store_true",
    )
    parser.add_argument(
        "--vrep-host",
        dest="vrepHost",
        help="Host of VREP. Default: %s" % config["vrepHost"],
        type=str,
    )
    parser.add_argument(
        "--vrep-port",
        dest="vrepPort",
        help="Port of VREP. Default: %i" % config["vrepPort"],
        type=int,
    )
    parser.add_argument(
        "--vrep-scene",
        dest="vrepScene",
        help=("Scene to load in VREP. Default: " + config["vrepScene"]),
        type=str,
    )
    parser.add_argument(
        "--rostopic-name",
        dest="rostopicName",
        help=("Topic name for ROS. Default:" + config["rostopicName"]),
        type=str,
    )
    parser.add_argument(
        "-j",
        "--joint-state-name",
        dest="jointStateName",
        help=(
            "Name of the joint state topic. PLEASE NOTE: "
            + "A lot of other nodes assume the joint state "
            + "note to be at /joint_states, so be careful "
            + "when changing it. Default: %s"
        )
        % config["jointStateName"],
        type=str,
    )
    parser.add_argument(
        "-f",
        "--fake-execution",
        dest="fake",
        help=("Publish fake joint states instead of real " + "joint states"),
        action="store_false",
    )
    parser.add_argument(
        "-p",
        "--pyrep",
        dest="pyrep",
        help=("Use pyrep instead of vrep remote api"),
        action="store_true",
    )
    parser.add_argument(
        "--headless",
        dest="headless",
        help=("Run vrep in headless mode (requires pyrep)"),
        action="store_true",
    )

    args = parser.parse_known_args()[0]
    if args.robotMotorFile:
        config["robotMotorFile"] = args.robotMotorFile
    config["vrep"] = args.vrep
    if args.vrepHost:
        config["vrepHost"] = args.vrepHost
    if args.vrepPort:
        config["vrepPort"] = args.vrepPort
    if args.vrepScene:
        config["vrepScene"] = args.vrepScene
    if args.rostopicName:
        config["rostopicName"] = args.rostopicName
    if args.jointStateName:
        config["jointStateName"] = args.jointStateName
    config["fakeExecution"] = args.fake
    config["pyrep"] = args.pyrep
    config["headless"] = args.headless

    # Set logging setting
    loggingLevel = logging.INFO
    try:
        loggingLevel = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "CRITICAL": logging.CRITICAL,
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "critical": logging.CRITICAL,
        }[args.logLevel]
    except KeyError:
        sys.stderr.write("LOGGING ERROR: Unknown log level %s\n" % args.logLevel)
        pass

    logger = logging.getLogger(__name__)
    logger.setLevel(loggingLevel)
    handler = RosLoggingHandler()
    logger.addHandler(handler)

    rosConnection = NicoRosMotion(config)

    rospy.spin()

    del rosConnection

    rospy.loginfo("test")
