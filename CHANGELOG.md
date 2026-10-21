# 更新记录

## 2026-09-24

- 将公开训练参数改为通用场景命名，并将测试子命令统一为 `test.py scene`。
- 新增通用场景配置 schema，同时继续兼容旧 EMD v2 配置和 checkpoint 字段。
- 整理 MID360、RS32 的便携训练与真实场景测试 YAML 示例。
- 重写中英文 README，说明相对原版 CMRNext 的数据、扰动和使用方式。

## 2026-07-31

- 建立不包含旧历史的 EMD/KITTI 训练与评估代码基线。
- 移除旧 Module5 和 OB3D 数据路径、训练脚本、切分脚本及缓存残留。
- 保留 EMD 权威测试划分、KITTI 固定左右相机外参测试，以及明确请求时使用的
  原始 CMRNext KITTI 随机扰动基准。
- 将训练后评估统一到 `test.py`，共享实现保留在
  `evaluate_flow_calibration.py`，并补充对应回归测试。
- 将项目代理说明 `AGENTS.md` 完整改为中文，方便后续追加项目要求。
- 删除旧的 `CLAUDE.md`，避免与 `AGENTS.md` 形成重复或冲突的操作说明。
