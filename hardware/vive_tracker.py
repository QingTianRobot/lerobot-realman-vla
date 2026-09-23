#!/usr/bin/env python3
"""
Vive Tracker OpenVR 接口

通过 OpenVR API 读取 HTC Vive Tracker 的实时位姿（位置+姿态）。
支持多个 Tracker 同时连接，通过序列号区分。

依赖:
  pip install openvr transformations

参考: https://github.com/TriadSemi/triad_openvr
"""

import time
import sys
import openvr
import math
import numpy as np

try:
    from tf_transformations import euler_from_matrix  # ROS 提供 (tf-transformations)
except ImportError:
    try:
        # PyPI 上的 Gohlke 原版, API 与 tf_transformations 相同 (提供 euler_from_matrix)
        from transformations import euler_from_matrix
    except ImportError:
        print("需要安装: pip install transformations  (或 ROS 的 tf-transformations)")
        raise


def convert_to_euler(pose_mat):
    """将 OpenVR 3×4 位姿矩阵转换为 [x, y, z, yaw, pitch, roll]"""
    mat = np.array([
        [pose_mat[0][0], pose_mat[0][1], pose_mat[0][2], pose_mat[0][3]],
        [pose_mat[1][0], pose_mat[1][1], pose_mat[1][2], pose_mat[1][3]],
        [pose_mat[2][0], pose_mat[2][1], pose_mat[2][2], pose_mat[2][3]],
        [0, 0, 0, 1]
    ])
    roll, pitch, yaw = euler_from_matrix(mat, axes='sxyz')
    roll = roll * 180 / math.pi
    pitch = pitch * 180 / math.pi
    yaw = yaw * 180 / math.pi
    x = pose_mat[0][3]
    y = pose_mat[1][3]
    z = pose_mat[2][3]
    return [x, y, z, yaw, pitch, roll]


def convert_to_quaternion(pose_mat):
    """将 OpenVR 3×4 位姿矩阵转换为 [x, y, z, w, qx, qy, qz]"""
    r_w = math.sqrt(abs(1 + pose_mat[0][0] + pose_mat[1][1] + pose_mat[2][2])) / 2
    r_x = (pose_mat[2][1] - pose_mat[1][2]) / (4 * r_w)
    r_y = (pose_mat[0][2] - pose_mat[2][0]) / (4 * r_w)
    r_z = (pose_mat[1][0] - pose_mat[0][1]) / (4 * r_w)
    x = pose_mat[0][3]
    y = pose_mat[1][3]
    z = pose_mat[2][3]
    return [x, y, z, r_w, r_x, r_y, r_z]


class vr_tracked_device:
    """单个 VR 追踪设备"""

    def __init__(self, vr_obj, index, device_class):
        self.device_class = device_class
        self.index = index
        self.vr = vr_obj

    def get_serial(self):
        """获取设备序列号"""
        return self.vr.getStringTrackedDeviceProperty(
            self.index,
            openvr.Prop_SerialNumber_String
        )

    def get_model(self):
        """获取设备型号"""
        return self.vr.getStringTrackedDeviceProperty(
            self.index,
            openvr.Prop_ModelNumber_String
        )

    def get_battery_percent(self):
        """获取电池电量"""
        return self.vr.getFloatTrackedDeviceProperty(
            self.index,
            openvr.Prop_DeviceBatteryPercentage_Float
        )

    def get_pose_euler(self):
        """获取位姿 [x, y, z, yaw, pitch, roll]"""
        pose = self.vr.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0,
            openvr.k_unMaxTrackedDeviceCount
        )
        if pose[self.index].bPoseIsValid:
            return convert_to_euler(pose[self.index].mDeviceToAbsoluteTracking)
        return None

    def get_pose_quaternion(self):
        """获取位姿 [x, y, z, w, qx, qy, qz]"""
        pose = self.vr.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0,
            openvr.k_unMaxTrackedDeviceCount
        )
        if pose[self.index].bPoseIsValid:
            return convert_to_quaternion(pose[self.index].mDeviceToAbsoluteTracking)
        return None


class triad_openvr:
    """OpenVR 设备管理器"""

    def __init__(self):
        self.vr = openvr.init(openvr.VRApplication_Other)
        self.devices = {}
        self._discover_devices()

    def _discover_devices(self):
        """发现所有已连接的 VR 设备"""
        for i in range(openvr.k_unMaxTrackedDeviceCount):
            device_class = self.vr.getTrackedDeviceClass(i)
            if device_class == openvr.TrackedDeviceClass_Invalid:
                continue

            device_name = {
                openvr.TrackedDeviceClass_HMD: "hmd",
                openvr.TrackedDeviceClass_Controller: "controller",
                openvr.TrackedDeviceClass_GenericTracker: "tracker",
                openvr.TrackedDeviceClass_TrackingReference: "tracking_reference",
            }.get(device_class, f"device_{i}")

            # 处理同类型多设备
            count = sum(1 for k in self.devices if k.startswith(device_name))
            if count > 0:
                device_name = f"{device_name}_{count + 1}"

            self.devices[device_name] = vr_tracked_device(self.vr, i, device_class)

    def print_discovered(self):
        """打印所有发现的设备"""
        print("Discovered VR Devices:")
        for name, device in self.devices.items():
            serial = device.get_serial()
            model = device.get_model()
            print(f"  {name}: {model} (SN: {serial})")


if __name__ == "__main__":
    print("Initializing OpenVR...")
    v = triad_openvr()
    v.print_discovered()

    # 持续输出 Tracker 位姿
    trackers = {k: v for k, v in v.devices.items() if "tracker" in k}
    if not trackers:
        print("No tracker found!")
        sys.exit(1)

    print(f"\nTracking {len(trackers)} tracker(s)... (Ctrl+C to stop)")
    try:
        while True:
            for name, tracker in trackers.items():
                pose = tracker.get_pose_euler()
                if pose:
                    print(f"\r{name}: x={pose[0]:.3f} y={pose[1]:.3f} z={pose[2]:.3f} "
                          f"yaw={pose[3]:.1f} pitch={pose[4]:.1f} roll={pose[5]:.1f}",
                          end='')
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nDone.")
