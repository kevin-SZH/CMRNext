# CMRNext 通用场景外参标定扩展

**语言：** [English](README.md) · 中文

本仓库是 [CMRNext](https://github.com/robot-learning-freiburg/CMRNext) 的非官方修改版，用于在用户显式选择的场景目录上训练和评估相机-LiDAR 外参标定。主要新增 MID360 与 RS32 传感器示例、固定外参与逐帧外参训练数据、多种在线扰动方式、确定性评估和可审计产物。

原版 CMRNext 的论文、模型、定位代码和实验结论属于上游作者。本修改版产生的行为和结果不代表原作者。

## 相对原版 CMRNext 的改动

| 范围 | 原版 CMRNext | 本仓库 |
|---|---|---|
| 标定训练数据 | KITTI、Argoverse 1、PandaSet 加载器 | 显式场景目录，可配置相机与 LiDAR 几何 |
| 传感器示例 | 公开自动驾驶数据集 | Livox MID360 与 RoboSense RS32 |
| 外参目标 | 围绕数据集标定随机扰动 | 存储 `T0`/逐帧 `Ti`、固定 `T0`，或围绕每帧 `Ti` 在线扰动 |
| 几何约定 | 原版数据集相关行为 | `delta = Ti @ inverse(T0)`，且 `delta @ T0 = Ti` |
| 评估入口 | 直接调用评估器 | `python test.py`，支持通用场景、KITTI 固定相机和原版 KITTI 扰动 |
| 可复现性 | 随数据集实现而定 | 显式场景、seed、dry-run、日志、checkpoint 阶段映射和 `errors.json` |

训练统一通过 `python train_calibration.py` 启动，并使用下文的通用场景数据结构。KITTI、Argoverse 和 PandaSet 代码仍保留给原版推理或定位功能使用，但当前训练入口不接受这些原始数据集。


## 安装

项目需要带 CUDA 的 OpenCV 和自定义 PnP 扩展，建议使用 Docker。

~~~bash
docker build . -t cmrnext-scene
docker run --gpus=all --shm-size=2g -it \
  -v /path/to/data:/data \
  -v /path/to/checkpoints:/root/CMRNext/checkpoints \
  cmrnext-scene
~~~

当前镜像基于 CUDA 11.3、PyTorch 1.11 和 OpenCV 4.7，Dockerfile 尚未适配 RTX 50 系列。手动安装请参考 [requirements.txt](requirements.txt)、[Dockerfile](Dockerfile) 和 [visibility_pkg](visibility_pkg/)。

## 场景数据结构

训练、验证和测试场景均由用户显式指定。目录名称、metadata 中的 split、manifest 和统计文件都不会改变场景用途。

~~~text
scene_001/
├── image_2/                  # 同步 RGB 图像，例如 000000.png
├── velodyne/                 # 同名 Nx4 float32 .bin 点云
├── calib.txt                 # 相机投影矩阵 P0 或 P2
├── nominal_extrinsic.txt     # 一个 LiDAR-to-camera T0，3x4 或 4x4
├── poses.txt                 # 每帧 LiDAR-to-camera Ti
└── residuals.txt             # 每帧 Ti @ inverse(T0)
~~~

图像和点云目录名可通过 `dataset.layout` 修改，文件名可通过 `dataset.files` 修改。图像与点云使用同名 stem 配对。平移单位为米，矩阵采用列向量，外参均把 LiDAR 点变换到相机坐标系。

~~~text
网络输入：T0 @ p_lidar
目标点云：Ti @ p_lidar
监督残差：delta = Ti @ inverse(T0)
重建关系：delta @ T0 = Ti
~~~

## 训练数据与扰动方式

三种模式使用相同场景结构：

1. **存储多外参加标称外参扰动**（`dynamic_nominal_augmentation`）：读取存储的 `T0`、逐帧 `Ti` 和 residual。每次训练访问在球内均匀采样平移和随机轴旋转，计算 `new_T0 = jitter @ stored_T0`，再重新计算 `Ti @ inverse(new_T0)`。平移半径按 `Ti` 平移尺度计算，并设置米制下限。
2. **固定单外参**（`fixed_t0_online_perturbation`）：要求全部帧满足 `Ti=T0`，存储 residual 为单位阵。训练时分别在对称均匀区间内采样 XYZ 平移分量和 XYZ 欧拉角分量，构造 `T_input = inverse(delta) @ Ti`。
3. **逐帧 Ti 在线扰动**（`per_ti_online_perturbation`）：每帧 `Ti` 保持为目标，生成输入时不使用存储的 `T0` 和 residual。每次访问重新逐轴采样 `delta`，构造 `T_input = inverse(delta) @ Ti`。

一个 YAML 只能启用一种在线扰动。存储多外参的正式评估不会重新采样 residual；固定 T0 的评估扰动由 seed 和样本索引确定，可重复复现。

## 示例 YAML

| 文件 | 用途 |
|---|---|
| `config/train_mid360_multi_extrinsic.yaml` | MID360 存储多外参，并对 `T0` 在线小扰动 |
| `config/train_mid360_multi_extrinsic_online.yaml` | MID360 逐帧 `Ti`，直接在线逐轴扰动 |
| `config/train_mid360_single_extrinsic.yaml` | MID360 固定 `T0`，在线逐轴扰动 |
| `config/train_rs32_multi_extrinsic.yaml` | RS32 存储多外参，并对 `T0` 在线小扰动 |
| `config/train_rs32_single_extrinsic.yaml` | RS32 固定 `T0`，在线逐轴扰动 |
| 训练 YAML 文件 | 评估时直接使用对应训练 YAML，并用 `--scene-paths` 覆盖场景 |

示例统一使用 `schema_version: cmrnext_scene_calibration_v1`。加载器仍接受旧的 `cmrnext_emd_training_v2`，保证已有配置可读。正式运行前应修改 `dataset.root`、场景列表、相机内参和传感器采集参数。

## 训练示例

所有训练先加 `--dry-run`，检查解析后的场景、几何、seed、输出目录、图像尺寸和扰动模式。dry-run 不加载样本，也不启动 CUDA。

### MID360 逐帧 Ti 在线扰动

~~~bash
python train_calibration.py \
  --config config/train_mid360_multi_extrinsic_online.yaml \
  --level perturbed_t0 \
  --img_shape 576 768 \
  --savemodel checkpoints --run_name cmrnext_ti_mid360_stage2 \
  --epochs 300 --batch_size 4 --num_worker 4 \
  --seed 42 --gpu -1 --gpu_count 4 --val_every 0 \
  --dry-run
~~~

### RS32 固定单外参

~~~bash
python train_calibration.py \
  --config config/train_rs32_single_extrinsic.yaml \
  --level fixed_baseline \
  --img_shape 576 768 \
  --savemodel checkpoints --run_name rs32_fixed_t0 \
  --epochs 300 --batch_size 4 --num_worker 4 \
  --seed 42 --gpu 0 --gpu_count 1 \
  --dry-run
~~~

确认报告后删除 `--dry-run`。命令行的 `--train-scene-paths`、`--val-scene-paths` 和 `--test-scene-paths` 会分别整组覆盖 YAML 列表。validation 为空时必须显式使用 `--val_every 0`。

训练产物位于 `<savemodel>/<run_name>/`，包括训练日志、TensorBoard events、滚动 checkpoint 和 last checkpoint。TensorBoard 会通过项目根目录下的 `tf-logs/<run_name>` 链接发现每次训练；需要时可用 `--tb_mount_root` 覆盖服务监听目录。使用 `--weights ... --resume` 恢复 optimizer 和 epoch；改变总 epoch 数时增加 `--rebuild_scheduler true`。

## 评估示例

多个 checkpoint 按级联顺序填写，`--start-stage` 表示第一个 checkpoint 的阶段编号。

`scene` 是显式场景评估所必需的子命令，用于选择 `test.py` 后面的评估模式；KITTI 评估则使用 `kitti` 或 `kitti-perturbed` 子命令。

### 显式场景评估

~~~bash
python test.py scene \
  --config config/train_mid360_multi_extrinsic.yaml \
  --level perturbed_t0 \
  --weights checkpoints/stage1.tar checkpoints/stage2.tar \
  --start-stage 1 --num-workers 4 --seed 42 --gpu 1 \
  --errors-json evaluation_reports/mid360/ti_2stage_errors.json \
  --log-file evaluation_reports/mid360/ti_2stage_evaluation.log \
  --dry-run
~~~

只有 YAML 提供非空的 `dataset.scenes.test` 列表时，才可以省略 `--scene-paths`。

### KITTI 左右相机固定物理外参

相机 `2` 是校正后的左相机，相机 `3` 是校正后的右相机。目标相机提供图像、内参和真实 `Ti`，另一相机提供固定物理 `T0`，两者不能相同。

~~~bash
python test.py kitti \
  --config config/train_mid360_multi_extrinsic.yaml \
  --data-root /data/KITTI/sequences --sequences 00 \
  --target-cam 2 --nominal-cam 3 \
  --weights checkpoints/model/last_iter_checkpoint.tar \
  --start-stage 1 --num-workers 4 --seed 42 --gpu 0 \
  --errors-json evaluation_reports/kitti_fixed/errors.json \
  --log-file evaluation_reports/kitti_fixed/evaluation.log \
  --dry-run
~~~

只有需要复现原版 CMRNext 随机扰动基准时才使用 `test.py kitti-perturbed`。论文尺度默认值为 `--max-t 1.5`、`--max-r 20`、`--seed 42`。

## 更换 LiDAR 时修改哪些参数

YAML 必须描述离线点云实际使用的传感器和累计方式：

| 字段 | 含义 |
|---|---|
| `rings` | 物理通道数，或数据生成时使用的虚拟分组数 |
| `points_per_ring` | 每个通道或分组对应的点数/发射束数 |
| `vertical_fov_deg` | 最小和最大垂直束角 |
| `repeated_pattern` | 旋转式重复扫描设为 `true`；MID360 等非重复扫描设为 `false` |
| `scan_frequency_hz` | 物理扫描频率 |
| `beams_per_scan` | 一个扫描周期的发射束数 |
| `beam_rate_hz` | 发射束率，必须等于 `beams_per_scan * scan_frequency_hz` |
| `accumulation_duration_s` | 每个离线点云已经累计的时间窗口 |

同时修改相机的 `width`、`height`、`fx`、`fy`、`cx`、`cy` 和 `skew`。`preprocessing.resize_hw` 控制在线 resize，内参会随图像同步缩放。加载器只审计采集参数，不会在线重新累计点云；修改 `accumulation_duration_s` 后必须重新生成点云，或切换到按该时长生成的场景。

旋转式多线 LiDAR 通常设 `repeated_pattern: true`，每个点云使用一个完整扫描周期；非重复扫描 LiDAR 设为 `false`，并填写实际束率和离线累计时间。

## 上游、引用与许可证

- 原始 CMRNext 论文代码：https://github.com/robot-learning-freiburg/CMRNext
- 项目主页：https://cmrnext.cs.uni-freiburg.de/
- 论文：*CMRNext: Camera to LiDAR Matching in the Wild for Localization and Extrinsic Calibration*
- arXiv：https://arxiv.org/abs/2402.00129
- IEEE T-RO：https://ieeexplore.ieee.org/document/10908048

本仓库沿用上游 [GNU GPLv3](LICENSE)。分发本修改版时请保留上游版权和许可证声明。

公开配置中的 `/path/to/dataset_root` 和 `scene_001` 是占位示例。训练前请替换成自己的数据根目录与显式场景路径。
