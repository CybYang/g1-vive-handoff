# VIVE 双手、Tracker 与 Inspire G2 重定向迁移交接

## 1. 交接范围和最终结构

本压缩包只包含 G1 PC2 需要运行的代码，不包含 Unity 工程，也不包含
SAPIEN、MediaPipe、RealSense、相机视频服务或 G1 手臂 IK/行走控制器。

已经包含的完整链路是：

```text
VIVE Focus Vision
  └─ UDP JSON：左/右 OpenXR 26 关键点 + Tracker 位姿
       ├─ G1 PC2:5005 → 左手独立进程
       │    → 协议校验 → 21点选择 → Unity/OpenXR坐标处理
       │    → 左手 V2 DexRetargeting/SLSQP
       │    → 13维 URDF关节角 → 左手 angleSet 映射
       │    → 左 USB-RS485 → 左 Inspire G2
       └─ G1 PC2:5006 → 右手独立进程
            → 同一流程 → 右 USB-RS485 → 右 Inspire G2
```

每个 UDP 包中仍保留 Tracker：ID、有效性、tracking state、三维位置和四元数。
当前灵巧手进程只解析并保留这些字段，不使用 Tracker 控制 G1 手臂。后续 G1
逆运动学模块应直接复用 `scripts/vive_focus_udp_receiver.py` 的
`ViveFocusFrame.trackers`，不要再定义一套不兼容的网络协议。

左右手使用两个进程，不是一个 26 自由度联合优化：

```text
左手 13维 SLSQP：独立初值、滤波和求解状态
右手 13维 SLSQP：独立初值、滤波和求解状态
```

这样一只手求解变慢不会阻塞另一只手，也适合 Orin NX 的多核 CPU。

## 2. 已经完成的算法状态

迁移版使用当前 V2，而不是最初的原版 DexPilot：

- 将 VIVE/OpenXR 的 26 个关节选择成重定向需要的 21 点；
- 消除 Unity 世界平移和朝向，构造腕部局部坐标；
- 左右手分别使用镜像一致的坐标变换；
- 左手任务参考点使用 `left_hand_base`；右手 `right_hand_palm` 与 base
  原点重合，继续使用 `right_hand_palm`；
- 在原版 DexPilot 任务损失上增加四指 MCP/PIP 软参考；
- 增加食指/中指侧摆参考、动态权重和侧摆耦合；
- 增加相邻四指的近似碰撞距离损失；
- 修复 SLSQP 返回的目标函数值没有包含时间平滑、但梯度包含该项的问题；
- 拇指采用分离的 CMC/IP 软参考；默认 CMC1/yaw 额外权重仍为 0；
- 曾尝试的“握拳/多指捏合状态机”已经回退，没有进入迁移版；
- 实机输出有寄存器限位、每包最大步长、发送频率限制和健康检查。

已知问题：拇指的 URDF 预测和实机可达位置仍有差异，尤其 CMC2。不要仅通过
继续提高损失权重解决。若后续采用固定拇指抓握位姿，必须新增一个明确的运行模式，
先在实机上标定可达的三个 `angleSet` 值，并保留当前动态重定向模式作为备份。

## 3. G1 PC2 的已知基线

交接时记录的目标设备是：

```text
hostname: unitree-g1-nx4
NVIDIA Jetson Orin NX 16 GB, aarch64
Ubuntu 22.04.4, Python 3.10
ROS 2 Humble / CycloneDDS
历史有线地址: 192.168.123.164
历史无线地址: 10.30.120.120
```

历史 IP 不能作为当前事实。每次迁移或换网络后，在 G1 上重新执行：

```bash
hostname
ip -br address
```

本包与 ROS 2、DDS、Unitree SDK、G1 手臂控制器互相独立。灵巧手走 USB-RS485，
VIVE 数据走 UDP。

## 4. 解压和环境配置

### 4.1 校验与解压

将压缩包和同名 `.sha256` 文件复制到 G1 后：

```bash
sha256sum -c inspire_g2_vive_g1_handoff_YYYYMMDD_HHMMSS.tar.gz.sha256
tar -xzf inspire_g2_vive_g1_handoff_YYYYMMDD_HHMMSS.tar.gz
cd inspire_g2_vive_g1_handoff
python3 scripts/verify_g1_package.py
```

压缩包不包含 `.venv`。本机 `.venv` 是 x86_64，绝对不能复制到 aarch64
Jetson 上使用。

### 4.2 系统准备

确认 Python 和 venv 支持：

```bash
python3 --version
uname -m
sudo apt update
sudo apt install -y python3-venv python3-pip libgomp1
```

预期为 Python 3.10 和 `aarch64`。

### 4.3 建立隔离环境

直接运行：

```bash
./scripts/setup_g1_env.sh
```

脚本会在本目录建立独立 `.venv`，安装 aarch64 wheels，随后运行双手合成输入
自检。它不会修改已有 `g1ik` 环境，也不会打开串口。

采用独立环境的原因：交接时 `g1ik` 使用 NumPy 1.26.4、Pinocchio 3.1.0，
而当前可直接安装的 aarch64 NLopt/PyTorch/Pinocchio wheels 使用另一套已验证
组合。把这些依赖强行装进 `g1ik` 有破坏 G1 手臂 IK 的风险。

依赖由 `requirements-g1.txt` 固定。主要版本为：

```text
Python 3.10
NumPy 2.2.6
SciPy 1.15.3
Torch 2.13.0
NLopt 2.11.0
Pinocchio (pin) 4.1.0
PySerial 3.5
```

如果 G1 无法联网，不要随意换版本。应在另一台能联网的 aarch64 Ubuntu
22.04/Python 3.10 主机上下载同一组 wheels，再离线复制。x86_64 wheels 不能用。

### 4.4 独立自检

任何时候都可以重新运行：

```bash
.venv/bin/python scripts/g1_preflight.py --strict-g1
```

成功标准：左右手都显示 `OK`，最后出现：

```text
PASS: no serial port opened; both G1 hand pipelines are self-contained.
```

## 5. 迁移后必须修改的地方

第一次执行环境脚本后会生成：

```text
config/g1_runtime.env
```

也可以手动创建：

```bash
cp config/g1_runtime.env.example config/g1_runtime.env
```

编辑它：

```bash
nano config/g1_runtime.env
```

### 5.1 UDP 监听地址和端口

正常保留：

```bash
BIND_ADDRESS=0.0.0.0
LEFT_UDP_PORT=5005
RIGHT_UDP_PORT=5006
```

`0.0.0.0` 是监听 G1 所有本地网卡，不是要填 Unity 或头显 IP。端口必须与
当前发包端一致，而且 5005/5006 不能被其他进程占用。

先将 `VIVE_SENDER_IP` 留空完成联调。确认终端状态行里的发送方地址后，建议填写：

```bash
VIVE_SENDER_IP=<头显当前Wi-Fi地址>
```

这样来自其他主机的同格式 UDP 包会被过滤。头显地址改变后需要同步更新。

### 5.2 左右 USB-RS485 稳定路径

先插入两个转换器，运行：

```bash
./scripts/list_g1_io.sh
```

不要长期依赖 `/dev/ttyUSB0` 和 `/dev/ttyUSB1`，它们在重启或重新插拔后可能
交换。优先把配置改成：

```bash
LEFT_SERIAL_PORT=/dev/serial/by-id/<左手转换器的稳定名字>
RIGHT_SERIAL_PORT=/dev/serial/by-id/<右手转换器的稳定名字>
```

如果两个转换器的 `by-id` 名称完全相同或没有唯一序列号，使用
`/dev/serial/by-path/`，并固定 USB 物理插口；或者后续添加 udev 规则。

### 5.3 手 ID

当前每只手使用独立 RS485 总线，所以通常保留：

```bash
LEFT_HAND_ID=1
RIGHT_HAND_ID=1
```

只有在手的寄存器 ID 实际修改过时才改。不要猜测。

### 5.4 速度和安全参数

第一次装到 G1 上保留：

```bash
HARDWARE_RATE_HZ=10
MAX_STEP_UNITS=2
HEALTH_CHECK_PERIOD_S=2
```

`MAX_STEP_UNITS=2` 表示每个已发送数据包最多变化 0.2°。确认右手和左手的
13 个通道方向、限位、线缆和机构干涉后，再逐级提高，例如 2 → 5 → 10 → 20。
不要为了追求速度直接跳到大值。

### 5.5 可选 CPU 绑定

默认先留空：

```bash
LEFT_CPUSET=
RIGHT_CPUSET=
```

观察状态行 `solve=...ms` 后，如果两个进程调度抖动明显，可设为不同空闲核心，
例如：

```bash
LEFT_CPUSET=2
RIGHT_CPUSET=3
```

不要占用 G1 全身控制器的实时核心；应先检查其现有 affinity/服务配置。

## 6. 推荐验收顺序

必须严格从只读到运动执行。不要直接启动双手实机。

### 6.1 第一步：环境自检，不接手也可以

```bash
.venv/bin/python scripts/g1_preflight.py --strict-g1
```

### 6.2 第二步：只检查 UDP、双手关键点和 Tracker

先确保端口没有被实机程序占用：

```bash
./scripts/list_g1_io.sh
```

检查左端口：

```bash
./scripts/run_g1_udp_check.sh left
```

停止后检查右端口：

```bash
./scripts/run_g1_udp_check.sh right
```

这是纯网络诊断，不打开串口。正常状态行应该看到：

```text
left=tracked 26/26 valid
right=tracked 26/26 valid
trackers=[id=... pose=valid ...]
rx=... Hz
```

如果 `trackers=[]`，表示当前包没有 Tracker 字段；手指重定向仍可运行，但 G1
末端位姿模块没有输入。

### 6.3 第三步：用真实 VIVE 数据干跑完整算法

这一步会运行真实的坐标处理、V2 SLSQP 和 13 维 `angleSet` 映射，但绝不打开
串口：

```bash
./scripts/run_g1_left_dry.sh
```

停止后：

```bash
./scripts/run_g1_right_dry.sh
```

观察 `rx`、`solve`、`invalid`、`coalesced` 和输出的 13 个 `angleSet` 数值。这样可以
在机器人不运动的前提下确认 aarch64 环境确实跑通了当前完整算法。

### 6.4 第四步：只读检查左右灵巧手

先分别测试，不发送角度命令：

```bash
.venv/bin/python tools/inspire_g2_readonly_probe.py \
  --port "$LEFT_SERIAL_PORT" --hand-id 1 --execute
```

```bash
.venv/bin/python tools/inspire_g2_readonly_probe.py \
  --port "$RIGHT_SERIAL_PORT" --hand-id 1 --execute
```

注意：上面的环境变量只有 `source config/g1_runtime.env` 后才能在当前 shell 中
使用；也可以直接填写稳定串口路径。检查温度、错误码、状态码和实际角度。

如果当前账户没有串口权限：

```bash
groups
sudo usermod -aG dialout "$USER"
```

之后必须注销重新登录，不能只开一个新终端。

### 6.5 第五步：右手单独低速实机

机器人周围清空，线缆留有余量，操作员保持急停可用，然后：

```bash
./scripts/run_g1_right_live.sh RUN_LIVE_RETARGETING
```

检查每个手指及拇指通道的方向。停止使用 `Ctrl+C`。

### 6.6 第六步：左手单独低速实机

```bash
./scripts/run_g1_left_live.sh RUN_LIVE_RETARGETING
```

同样逐通道验证。

### 6.7 第七步：双手两个独立进程

两边均已单独验收后：

```bash
./scripts/run_g1_both_live.sh RUN_LIVE_RETARGETING
```

它会启动两个独立进程，并分别写入：

```text
log/g1_left_live.log
log/g1_right_live.log
```

任一进程退出时，包装脚本会停止另一进程。也可以手动开两个 SSH 终端，分别
运行左右脚本，便于调试。

## 7. 运行状态如何判断

实机状态行主要字段：

```text
rx=23.5Hz       收到的最新 UDP 帧率
solve=18.0ms    当前一只手的 SLSQP 耗时
hold_age=...    距离上次有效手部关键点的时间
accepted=...    成功处理帧数
invalid=...     手未跟踪/关键点无效帧数
errors=...      求解或映射错误数
sent=...        已发送 angleSet 次数
coalesced=...   为降低延迟而丢弃的旧 UDP 包数
```

建议标准：

- `rx` 稳定且两端口接近；
- `solve` 大部分低于输入帧间隔；
- `invalid/errors` 不持续快速增长；
- `hold_age` 在手被跟踪时保持较小；
- `coalesced` 偶尔增加可以接受，持续快速增加表示求解跟不上；
- 视频服务和 G1 控制器同时运行后，再比较这些数值是否恶化。

程序遇到无效手数据时不会生成新角度目标，实机会保持上一次发送的位置；这不是
自动张手或急停。整机安全必须由上层控制和现场急停保证。

## 8. 后续常改文件索引

### 网络协议与 Tracker

```text
scripts/vive_focus_udp_receiver.py
```

负责 UDP JSON 校验、左右 26 点和 Tracker。协议当前为 `v=1`，坐标为
`unity_world`，单位为米。若后续修改包结构，要同时升级版本并保留旧包兼容或明确
拒绝，不能静默改变字段含义。

Tracker 四元数目前按收到的 4 个数原样保存。将它用于 G1 IK 前，必须在末端位姿
模块中明确 Unity 四元数排列和左右手坐标变换，不要直接照搬“手部局部关键点”的
变换，因为 Tracker 表示世界/跟踪空间位姿，语义不同。

### OpenXR 手部坐标处理

```text
scripts/vive_openxr_hand.py
```

负责 26→21 选择、Unity 左手坐标转右手坐标、减去 wrist、构造手部局部基。若出现
左右镜像、手掌翻转或整体方向错误，先检查这里和输入有效性，不要先调 SLSQP 权重。

### 当前 V2 算法

```text
scripts/inspire_g2_vive_runtime.py
scripts/inspire_g2_pose_adapter.py
src/dex_retargeting/optimizer.py
src/dex_retargeting/retargeting_config.py
```

- `runtime.py`：每帧总流程和 V2 参数传递；
- `pose_adapter.py`：人体关节角、软参考、动态门控、拇指参考；
- `optimizer.py`：DexPilot、关节/耦合/碰撞/时间损失及解析梯度；
- `retargeting_config.py`：从配置构建 Pinocchio/NLopt 求解器。

修改目标函数后必须运行梯度有限差分测试。SLSQP 要求“返回的目标函数值”和“返回
梯度”描述同一个函数；只改其中一个会导致收敛异常。

### 左右 URDF 和任务参考点

```text
custom_assets/inspire_g2_hand/inspire_g2_hand_left.urdf
custom_assets/inspire_g2_hand/inspire_g2_hand_right.urdf
src/dex_retargeting/configs/teleop/inspire_g2_hand_left_dexpilot.yml
src/dex_retargeting/configs/teleop/inspire_g2_hand_right_dexpilot.yml
```

不要只按 link 名字假设左右完全对称。当前左 palm 相对 base 有 29.5 mm 偏移，右
palm 与 base 重合，因此左配置特意使用 base。更换厂商 URDF 后要重新检查 fixed
origin、关节轴、限位、mimic 关系和指尖 link。

### URDF 关节角到实机寄存器

```text
scripts/inspire_g2_left_mapping.yaml
scripts/inspire_g2_right_mapping.yaml
scripts/inspire_g2_hardware.py
tools/demo_serial_064_safe.py
```

左右 YAML 当前主要使用 URDF/手册端点形成分段线性映射。实机与仿真差异明显时，
应该在 YAML 中加入测得的中间锚点 `[URDF弧度, angleSet值]`，而不是改变网络坐标。
右手仍属于标称映射，必须通过低速实机逐关节验证。

### 默认运行参数

```text
config/g1_runtime.env
```

部署相关的端口、串口、手 ID、速度、步长和 CPU affinity 都在这里调整，不需要改
Python 源码。算法权重当前由 `inspire_g2_vive_hardware.py` 的参数默认值管理；实验
参数可以追加在启动命令后。

## 9. 测试

完整迁移包测试：

```bash
./scripts/test_g1_package.sh
```

重点覆盖：

- UDP 与 Tracker 协议校验；
- OpenXR 坐标与左右镜像；
- V2 目标函数和解析梯度；
- 左右 URDF/mimic；
- 13 维硬件映射与限位；
- 启动参数和串口安全确认；
- 左右 SLSQP 独立/并发求解。

所有自动测试均不得打开真实串口。真实硬件只通过显式
`RUN_LIVE_RETARGETING` 启动脚本打开。

## 10. 常见故障

### 收不到 UDP

```bash
ip -br address
ss -lunp | grep -E ':5005|:5006'
sudo ufw status
```

确认 G1 与头显在可互通网络，端口没有被旧进程占用。若启用 UFW，仅开放所需的
UDP 5005/5006，并限制为可信局域网来源。

### 只有一只手有数据

分别运行 `run_g1_udp_check.sh left/right`。两个端口都应收到“同一个完整双手包”；
单手进程只选择自己的 `frame.left` 或 `frame.right`。

### 串口打不开

检查 `config/g1_runtime.env`、`/dev/serial/by-id`、dialout 权限以及是否有另一个进程
占用设备：

```bash
fuser -v /dev/ttyUSB0
```

### `ModuleNotFoundError` 或二进制导入错误

确认使用本包 `.venv/bin/python`，不是系统 Python 或 `g1ik`。不要复制本机 x86
`.venv`。重新运行 `setup_g1_env.sh` 和 `g1_preflight.py --strict-g1`。

### 延迟越来越大

接收器会只处理最新包，不应该排队回放。检查 `solve` 和 `coalesced`；保持数值库
线程数为 1，必要时给左右进程绑定不同空闲核心。视频编码和 G1 全身控制器上线后
重新测量，不能只凭本地工作站结果判断 Orin 性能。

### 仿真好、实机捏合不到

优先比较硬件 `angleSet/angleAct` 和映射锚点。URDF 中指尖接触不代表实机能到达。
尤其检查拇指 CMC2/IP，不要无限增加软约束权重。

## 11. 本包不做的事情

- 不启动或控制 SAPIEN；
- 不读取 MediaPipe/相机关键点；
- 不修改 Unity；
- 不求解 G1 手臂 IK；
- 不把 Tracker 位姿直接发给 G1 控制器；
- 不控制 G1 行走或全身平衡；
- 不运行第一视角视频服务。

这些模块以后可以在 G1 PC2 上作为独立进程接入，但不能与当前灵巧手代码争用串口，
也不能让多个身体控制器同时写 G1 的相同关节。

## 12. 交接验收清单

- [ ] 压缩包 SHA256 校验通过；
- [ ] G1 显示 Python 3.10 / aarch64；
- [ ] `setup_g1_env.sh` 完成；
- [ ] 双手 preflight 通过；
- [ ] 记录 G1 当前 Wi-Fi IP；
- [ ] 5005/5006 都收到完整双手和 Tracker；
- [ ] 左右 USB-RS485 使用稳定设备路径；
- [ ] 左右手只读探测均通过；
- [ ] `MAX_STEP_UNITS=2` 下右手单独验收；
- [ ] `MAX_STEP_UNITS=2` 下左手单独验收；
- [ ] 双手进程同时运行并记录 solve/rx/sent；
- [ ] 与 G1 手臂 IK、全身控制器、视频服务同时运行后重新做性能和安全验收。
