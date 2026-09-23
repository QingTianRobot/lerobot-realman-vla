#!/usr/bin/env python3
"""
睿尔曼 RealMan RM65 机械臂只读自检

只做"连接 + 读状态 / 读信息"，**不下发任何运动、使能、运行模式指令**，
可安全地在采集/推理前反复运行，用于确认机械臂是否在线、健康。

分层检查:
  1. IP 冲突: 目标 IP 是否就是本机网卡地址 (是则 ping 通但 TCP 必拒绝)
  2. TCP:     目标 ip:port 是否可连接 (RM 服务在监听)
  3. SDK:     rm_create_robot_arm 句柄有效 + rm_get_current_arm_state code==0
  4. 信息:    rm_get_robot_info / rm_get_arm_software_info (型号/固件)

⚠️ 同网段可能有多台同型号 RM65 (SN 读不出、型号固件相同), 无法靠 SN 区分;
   用 --scan 列出所有开 8080 的候选, 再结合关节角与物理停放姿态肉眼比对锁定。

依赖:
  pip install robotic-arm   (RM_API2, 提供 Robotic_Arm.rm_robot_interface)

自检:
  python hardware/realman_arm.py [--arm-ip 192.168.5.123] [--arm-port 8080]
  python hardware/realman_arm.py --scan          # 扫网段列出候选机械臂
  python hardware/realman_arm.py --scan --all    # 扫描并逐台读状态
  python hardware/realman_arm.py --read-init           # 读当前位姿→打印 ROBOT_INIT 粘贴行
  python hardware/realman_arm.py --read-init --write   # 读当前位姿→直接写入 collect_data.py
"""

import os
import re
import socket
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor

DEFAULT_ARM_IP = "192.168.5.123"
DEFAULT_ARM_PORT = 8080
DEFAULT_SUBNET = "192.168.5"
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_COLLECT_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "collect_data.py")


def local_ips():
    """本机所有 IPv4 地址 (用于 IP 冲突自检)。"""
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True,
                             text=True, timeout=5)
        if out.stdout.split():
            return set(out.stdout.split())
    except Exception:
        pass
    try:
        return set(socket.gethostbyname_ex(socket.gethostname())[2])
    except Exception:
        return set()


def check_tcp(ip, port, timeout=2.0):
    """目标 ip:port 是否可 TCP 连接 (RM 服务在监听)。"""
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def probe_subnet(subnet=DEFAULT_SUBNET, port=DEFAULT_ARM_PORT, workers=64):
    """并发探测整个 /24 网段, 返回 port 开放的主机 IP 列表 (跳过本机)。"""
    mine = local_ips()
    cands = [f"{subnet}.{i}" for i in range(1, 255) if f"{subnet}.{i}" not in mine]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        flags = list(ex.map(lambda a: check_tcp(a, port, 1.0), cands))
    hits = [ip for ip, ok in zip(cands, flags) if ok]
    return sorted(hits, key=lambda s: int(s.rsplit(".", 1)[1]))


def read_arm(ip, port=DEFAULT_ARM_PORT):
    """只读连接机械臂并读取状态/机型/固件。返回 dict; 不产生任何运动。"""
    from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
    res = {"ip": ip, "connect": False, "state_code": None,
           "joint": None, "pose": None, "model": None, "product": None}
    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    try:
        handle = arm.rm_create_robot_arm(ip, port)
        if handle.id == -1:
            return res
        res["connect"] = True

        code, state = arm.rm_get_current_arm_state()
        res["state_code"] = code
        if code == 0:
            res["joint"] = [round(float(x), 3) for x in state.get("joint", [])]
            pose = state.get("pose")
            if pose:
                res["pose"] = [round(float(x), 4) for x in pose]
        try:
            _, info = arm.rm_get_robot_info()
            res["model"] = info.get("arm_model")
        except Exception:
            pass
        try:
            _, sw = arm.rm_get_arm_software_info()
            res["product"] = sw.get("product_version")
        except Exception:
            pass
    except Exception as e:  # noqa: BLE001 - 自检需吞掉 SDK 异常并汇报
        res["error"] = str(e)
    finally:
        try:
            arm.rm_delete_robot_arm()
        except Exception:
            pass
    return res


def self_test(ip, port):
    """对单台机械臂做分层只读自检, 返回是否健康。"""
    healthy = True
    print("=" * 50)
    print(f"  RealMan RM65 只读自检  target={ip}:{port}")
    print("=" * 50)

    # 1. IP 冲突
    if ip in local_ips():
        print(f"[!] IP 冲突: {ip} 是本机网卡地址 —— ping 会通(在 ping 自己)但 TCP 必拒绝。"
              f"请用 --scan 定位真实机械臂。")
        healthy = False

    # 2. TCP
    tcp_ok = check_tcp(ip, port)
    print(f"[{'✓' if tcp_ok else '!'}] TCP {ip}:{port} : "
          f"{'开放' if tcp_ok else '拒绝/不可达'}")
    if not tcp_ok:
        print(f"[i] 提示: 用 --scan 扫网段找开 {port} 的候选机械臂。")
        return False

    # 3-4. SDK 连接 + 状态 + 信息
    r = read_arm(ip, port)
    if not r["connect"]:
        print(f"[!] SDK 连接失败 (handle=-1){': ' + r.get('error','') if r.get('error') else ''}")
        return False
    print("[✓] SDK 连接 OK")

    if r["state_code"] == 0:
        print(f"[✓] 状态 code=0  关节角(°)={r['joint']}")
    else:
        print(f"[!] 读状态异常 code={r['state_code']}")
        healthy = False

    print(f"[i] 型号={r['model']}  产品/固件={r['product']}")
    print("-" * 50)
    print("结论: 机械臂正常 (只读自检通过)" if healthy else "结论: 机械臂异常, 见上方 [!] 项")
    return healthy


def write_robot_init(pose, target=None):
    """把 pose 原地写入 collect_data.py 的 ROBOT_INIT_POS / ROBOT_INIT_ORI 两行。

    只替换这两行的 np.array([...]) 内容; 定位不到就放弃修改 (不破坏文件)。
    """
    target = target or DEFAULT_COLLECT_SCRIPT
    if not os.path.isfile(target):
        print(f"[!] 目标文件不存在: {target}")
        return False
    with open(target, "r", encoding="utf-8") as f:
        src = f.read()

    pat_pos = re.compile(r"^ROBOT_INIT_POS = np\.array\(\[[^\]]*\]\)", re.M)
    pat_ori = re.compile(r"^ROBOT_INIT_ORI = np\.array\(\[[^\]]*\]\)", re.M)
    m_pos, m_ori = pat_pos.search(src), pat_ori.search(src)
    if not m_pos or not m_ori:
        print(f"[!] 在 {target} 未定位到 ROBOT_INIT_POS/ORI 行, 未修改")
        return False

    new_pos = f"ROBOT_INIT_POS = np.array([{pose[0]}, {pose[1]}, {pose[2]}])"
    new_ori = f"ROBOT_INIT_ORI = np.array([{pose[3]}, {pose[4]}, {pose[5]}])"
    src = pat_pos.sub(lambda _: new_pos, src, count=1)
    src = pat_ori.sub(lambda _: new_ori, src, count=1)
    with open(target, "w", encoding="utf-8") as f:
        f.write(src)

    print("-" * 50)
    print(f"[✓] 已写入 {os.path.relpath(target, _REPO_ROOT)}:")
    print(f"    {m_pos.group(0)}\n -> {new_pos}")
    print(f"    {m_ori.group(0)}\n -> {new_ori}")
    return True


def read_init(ip, port, write=False, target=None):
    """只读当前笛卡尔位姿; 默认打印可粘贴行, --write 时直接写入 collect_data.py。

    用途: 手动把机械臂拖到安全顺手的遥操起始位后运行本命令, 更新 scripts/collect_data.py
    里的 ROBOT_INIT_POS / ROBOT_INIT_ORI。全程不下发任何运动指令。
    """
    if not check_tcp(ip, port):
        print(f"[!] TCP {ip}:{port} 不可达 —— 机械臂没上电/IP 不对; 用 --scan 定位真实机械臂。")
        return False

    r = read_arm(ip, port)
    pose = r.get("pose")
    if not r["connect"] or r["state_code"] != 0 or not pose or len(pose) < 6:
        print(f"[!] 读取位姿失败: connect={r['connect']} code={r['state_code']} "
              f"pose={pose} {('err=' + r['error']) if r.get('error') else ''}")
        return False

    print("=" * 50)
    print(f"  当前笛卡尔位姿  target={ip}:{port}")
    print("=" * 50)
    print(f"[✓] pose [x,y,z, rx,ry,rz] (米/弧度) = {pose}")
    print(f"[i] 关节角(°) = {r['joint']}")

    if write:
        return write_robot_init(pose, target)

    print("-" * 50)
    print("把下面两行粘贴到 scripts/collect_data.py, 覆盖 ROBOT_INIT_POS / ROBOT_INIT_ORI:")
    print(f"ROBOT_INIT_POS = np.array([{pose[0]}, {pose[1]}, {pose[2]}])")
    print(f"ROBOT_INIT_ORI = np.array([{pose[3]}, {pose[4]}, {pose[5]}])")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="睿尔曼 RM65 机械臂只读自检 (不下发运动)")
    parser.add_argument("--arm-ip", type=str, default=DEFAULT_ARM_IP)
    parser.add_argument("--arm-port", type=int, default=DEFAULT_ARM_PORT)
    parser.add_argument("--subnet", type=str, default=DEFAULT_SUBNET,
                        help="--scan 时扫描的 /24 网段前缀")
    parser.add_argument("--scan", action="store_true",
                        help="扫网段列出所有开放机械臂端口的主机")
    parser.add_argument("--all", action="store_true",
                        help="配合 --scan: 对每台候选逐台读状态")
    parser.add_argument("--read-init", action="store_true",
                        help="读当前笛卡尔位姿, 打印可粘贴到 collect_data.py 的 ROBOT_INIT_POS/ORI 行")
    parser.add_argument("--write", action="store_true",
                        help="配合 --read-init: 直接把读到的位姿写入 --target 的 collect_data.py")
    parser.add_argument("--target", type=str, default=DEFAULT_COLLECT_SCRIPT,
                        help="--write 要修改的 collect_data.py 路径 (默认 scripts/collect_data.py)")
    args = parser.parse_args()

    if args.scan:
        hits = probe_subnet(args.subnet, args.arm_port)
        print(f"网段 {args.subnet}.0/24 开放 {args.arm_port} 的主机: {hits or '无'}")
        if args.all:
            for ip in hits:
                r = read_arm(ip, args.arm_port)
                print(f"  {ip}: connect={r['connect']} state_code={r['state_code']} "
                      f"joint={r['joint']} model={r['model']} product={r['product']}")
        raise SystemExit(0)

    if args.read_init:
        raise SystemExit(0 if read_init(args.arm_ip, args.arm_port,
                                        write=args.write, target=args.target) else 1)

    raise SystemExit(0 if self_test(args.arm_ip, args.arm_port) else 1)
