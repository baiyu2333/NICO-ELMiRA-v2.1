#!/usr/bin/env python3
"""Manual XL-320 motor ID movement test through the existing Motion wrapper.

This script is for supervised bench debugging only. It publishes Protocol 2
write commands to /nico/motion/xl320_cmd, so Motion.py must already be running.
It does not command arm trajectories or publish NICO arm joint goals.
"""

import argparse
import re
import time

import rosgraph
import rospy
from nicomsg.msg import sff


SAFE_MIN_DEG = -120.0
SAFE_MAX_DEG = 120.0


def parse_ids(value):
    ids = [int(match) for match in re.findall(r"\d+", str(value))]
    if not ids:
        raise argparse.ArgumentTypeError("Provide at least one motor id, e.g. 30 or 30,31")
    return ids


def clamp_deg(value):
    return max(SAFE_MIN_DEG, min(SAFE_MAX_DEG, float(value)))


def deg_to_raw(deg):
    raw = int((float(deg) + 150.0) / 300.0 * 1023.0)
    return max(0, min(1023, raw))


def publish_cmd(pub, motor_id, register, value, delay=0.03):
    msg = sff()
    msg.param1 = str(int(motor_id))
    msg.param2 = float(register)
    msg.param3 = float(value)
    pub.publish(msg)
    rospy.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description="Supervised XL-320 motor ID movement test")
    parser.add_argument("--ids", type=parse_ids, required=True, help="Motor IDs, e.g. 30 or 30,31")
    parser.add_argument("--low-deg", type=float, default=-80.0)
    parser.add_argument("--high-deg", type=float, default=80.0)
    parser.add_argument("--duration-sec", type=float, default=5.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--speed", type=int, default=120, help="XL-320 moving speed raw value")
    parser.add_argument("--release", action="store_true", help="Disable torque on tested IDs after the test")
    parser.add_argument("--confirm", action="store_true", help="Required: confirm robot workspace is clear")
    args = parser.parse_args()

    if not args.confirm:
        raise SystemExit("Refusing to run. Add --confirm after checking the robot workspace is clear.")

    ids = args.ids
    low_deg = clamp_deg(args.low_deg)
    high_deg = clamp_deg(args.high_deg)
    duration_sec = max(1.0, float(args.duration_sec))
    step_sec = max(0.2, float(args.step_sec))
    speed = max(1, min(1023, int(args.speed)))

    if not rosgraph.is_master_online():
        raise SystemExit(
            "ROS master is not running. Open the dashboard and click Launch ELMiRA first, "
            "or start roscore + Motion.py before running this test."
        )

    rospy.init_node("xl320_manual_id_test", anonymous=True, disable_signals=True)
    pub = rospy.Publisher("/nico/motion/xl320_cmd", sff, queue_size=10)
    deadline = time.time() + 3.0
    while pub.get_num_connections() == 0 and time.time() < deadline and not rospy.is_shutdown():
        rospy.sleep(0.05)
    if pub.get_num_connections() == 0:
        raise SystemExit("No subscriber on /nico/motion/xl320_cmd. Start Motion.py or Launch ELMiRA first.")

    print(f"Testing XL-320 IDs {ids}")
    print(f"Range: {low_deg:.1f} deg <-> {high_deg:.1f} deg for {duration_sec:.1f}s")
    print(f"Raw: {deg_to_raw(low_deg)} <-> {deg_to_raw(high_deg)}, speed={speed}")

    for motor_id in ids:
        publish_cmd(pub, motor_id, 24, 1)      # torque enable
        publish_cmd(pub, motor_id, 32, speed)  # moving speed

    end_time = time.time() + duration_sec
    target_high = False
    step_index = 0
    while time.time() < end_time and not rospy.is_shutdown():
        target_high = not target_high
        target_deg = high_deg if target_high else low_deg
        target_raw = deg_to_raw(target_deg)
        step_index += 1
        for motor_id in ids:
            publish_cmd(pub, motor_id, 30, target_raw)
        print(f"step {step_index}: IDs {ids} -> {target_deg:.1f} deg raw={target_raw}")
        time.sleep(step_sec)

    if args.release:
        for motor_id in ids:
            publish_cmd(pub, motor_id, 24, 0)
        print(f"Released torque on IDs {ids}")
    else:
        print("Torque left enabled. Use --release next time if you want torque disabled after test.")


if __name__ == "__main__":
    main()
