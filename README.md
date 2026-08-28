# G1 VIVE Hand Retargeting Handoff

面向 Unitree G1 PC2（Jetson Orin NX）的 VIVE Focus Vision 双手遥操作交接仓库。

本项目接收 Unity/OpenXR 通过 UDP 发送的左右手关键点和 VIVE Tracker 位姿，分别运行左右手 V2 DexRetargeting，并通过两路 USB-RS485 控制两只 Inspire G2 灵巧手。

> [!WARNING]
> 本项目可以向真实灵巧手发送运动指令。首次部署必须依次完成环境自检、UDP 检查、算法 dry-run、串口只读探测和单手低速验证。不要直接启动双手实机。

## 系统结构

```text
VIVE Focus Vision + Unity/OpenXR
  └─ UDP JSON：双手各 26 个关键点 + Tracker 位姿
       ├─ G1 PC2 :5005
       │    └─ 左手坐标处理 → 13 维 V2 SLSQP
       │         → angleSet 映射 → 左 USB-RS485 → 左 Inspire G2
       └─ G1 PC2 :5006
            └─ 右手坐标处理 → 13 维 V2 SLSQP
                 → angleSet 映射 → 右 USB-RS485 → 右 Inspire G2
```

左右手使用两个独立进程，各自维护求解初值、时间平滑和硬件输出状态。一只手求解变慢时不会直接阻塞另一只手。

Tracker 的 ID、有效性、tracking state、三维位置和四元数会被解析并保留，但当前仓库**不会**使用 Tracker 控制 G1 手臂。

## 当前包含内容

- VIVE/OpenXR 左右手 26 点与 Tracker UDP 接收和校验；
- 26→21 手部关键点选择、Unity/OpenXR 坐标转换和腕部局部坐标构造；
- 左右 Inspire G2 的 URDF、STL、关节配置和 13 维实机映射；
- 当前 V2 DexRetargeting/SLSQP 算法；
- 单手和双手 dry-run、实机启动脚本；
- G1 环境安装、部署预检、串口只读检查及自动测试。

V2 相比基础 DexPilot 增加了四指 MCP/PIP 软参考、食指/中指侧摆参考与动态门控、侧摆耦合、相邻手指近似碰撞损失和拇指 CMC/IP 参考，并修复了时间平滑项的目标函数值与解析梯度不一致问题。

## 不包含内容

本仓库不包含：

- Unity 工程；
- SAPIEN 仿真；
- MediaPipe、RealSense 或第一视角视频服务；
- G1 手臂逆运动学、全身控制和行走控制；
- Tracker 到 G1 末端位姿控制器的转换与发送。

## 目标环境

交接版本验证环境：

```text
Unitree G1 PC2
NVIDIA Jetson Orin NX 16 GB
Ubuntu 22.04 / aarch64
Python 3.10
```

项目会创建独立的 `.venv`，不会修改 G1 现有的 `g1ik`、ROS 2、DDS 或 Unitree SDK 环境。

## 快速部署

在 G1 PC2 上执行：

```bash
git clone https://github.com/CybYang/g1-vive-handoff.git
cd g1-vive-handoff

python3 scripts/verify_g1_package.py
./scripts/setup_g1_env.sh
```

安装脚本会：

1. 检查 Python 3.10 和系统架构；
2. 创建项目独立 `.venv`；
3. 安装 `requirements-g1.txt` 中固定的依赖；
4. 创建 `config/g1_runtime.env`；
5. 运行不会打开串口的 G1 双手预检。

随后编辑运行配置：

```bash
nano config/g1_runtime.env
```

重点确认：

```text
BIND_ADDRESS=0.0.0.0
LEFT_UDP_PORT=5005
RIGHT_UDP_PORT=5006
VIVE_SENDER_IP=
LEFT_SERIAL_PORT=/dev/serial/by-id/...
RIGHT_SERIAL_PORT=/dev/serial/by-id/...
LEFT_HAND_ID=1
RIGHT_HAND_ID=1
MAX_STEP_UNITS=2
```

`0.0.0.0` 表示监听 G1 的所有本地网卡，不要改成头显 IP。USB-RS485 应优先使用 `/dev/serial/by-id/` 或 `/dev/serial/by-path/` 的稳定路径。

完整配置和迁移说明见 [G1 迁移交接文档](docs/G1_MIGRATION_HANDOFF_ZH.md)。

## 推荐验收顺序

### 1. 环境自检

```bash
.venv/bin/python scripts/g1_preflight.py --strict-g1
```

### 2. 只检查 UDP、双手关键点和 Tracker

以下命令不会打开串口：

```bash
./scripts/run_g1_udp_check.sh left
./scripts/run_g1_udp_check.sh right
```

正常状态应包含：

```text
left=tracked 26/26 valid
right=tracked 26/26 valid
trackers=[id=... pose=valid ...]
```

### 3. 用真实 VIVE 数据 dry-run

这一步会运行完整坐标处理、V2 SLSQP 和 13 维映射，但不会打开串口：

```bash
./scripts/run_g1_left_dry.sh
./scripts/run_g1_right_dry.sh
```

### 4. 串口只读检查

先查看设备：

```bash
./scripts/list_g1_io.sh
```

再按照交接文档分别读取左右手状态。只读探测不会发送 `angleSet`。

### 5. 单手低速实机

确认关节方向、限位、线缆和急停后，先右手、再左手：

```bash
./scripts/run_g1_right_live.sh RUN_LIVE_RETARGETING
./scripts/run_g1_left_live.sh RUN_LIVE_RETARGETING
```

`RUN_LIVE_RETARGETING` 是必须显式输入的安全确认参数。缺少该参数时，实机启动脚本会拒绝打开串口。

### 6. 双手实机

只有左右手都完成单独验收后才运行：

```bash
./scripts/run_g1_both_live.sh RUN_LIVE_RETARGETING
```

日志写入：

```text
log/g1_left_live.log
log/g1_right_live.log
```

按 `Ctrl+C` 会停止两个进程；任一进程异常退出时，包装脚本也会停止另一进程。

## 测试

在完成环境安装后运行：

```bash
./scripts/test_g1_package.sh
```

测试覆盖 UDP/Tracker 协议、OpenXR 坐标处理、V2 目标函数及解析梯度、左右 URDF/mimic、13 维实机映射和双手独立求解。自动测试不会打开真实串口。

## 目录结构

```text
config/                         G1 端口、串口、速度和安全参数模板
custom_assets/inspire_g2_hand/ 左右手 URDF 与 STL
docs/                           完整中文迁移交接文档
scripts/                        UDP、重定向、硬件接口和启动脚本
src/dex_retargeting/            DexRetargeting 核心与手型配置
tests/                          协议、算法、映射及部署测试
tools/                          串口只读探测和安全调试工具
```

常改文件索引：

- 网络协议与 Tracker：`scripts/vive_focus_udp_receiver.py`
- OpenXR 手部坐标：`scripts/vive_openxr_hand.py`
- 每帧 V2 流程：`scripts/inspire_g2_vive_runtime.py`
- 人手到机器人手参考：`scripts/inspire_g2_pose_adapter.py`
- 目标函数与梯度：`src/dex_retargeting/optimizer.py`
- 左右实机映射：`scripts/inspire_g2_left_mapping.yaml`、`scripts/inspire_g2_right_mapping.yaml`

## 已知限制

- 拇指 CMC2 的 URDF 预测与实机可达位置仍存在差异；
- Tracker 尚未接入 G1 手臂 IK；
- 更换网络后需要重新确认 G1 和头显 IP；
- USB 转换器重新插拔后 `/dev/ttyUSB*` 编号可能交换；
- 与 G1 全身控制器、手臂 IK 和视频服务同时运行时，需要重新测量延迟与 CPU 占用。

## 来源与许可

DexRetargeting 核心基于 [dexsuite/dex-retargeting](https://github.com/dexsuite/dex-retargeting)，代码许可见 [LICENSE](LICENSE)。

`custom_assets/inspire_g2_hand/` 中的 Inspire G2 URDF/STL 来源于厂商资源包，仅将 ROS `package://` 网格路径改为项目内相对路径；相关模型文件的使用与再分发应同时遵循厂商条款。
