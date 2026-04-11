# RL-Based Arbitrary-Shaped Multi-Agent Path Finding

This repository is based on `rl4mapf` and is being extended toward arbitrary-shaped, multi-cell MAPF agents with rotation.

## Featured Algorithm

- CACTUS: Confidence-based Auto-Curriculum for Team Update Stability [1]

## Refactor Progress

| 状态 | 阶段 | 目标 | 主要改动文件 | 产出 | 预计 |
| --- | --- | --- | --- | --- | --- |
| 已完成 | 1 | 冻结建模规格 | 新增配置，整理常量 | 确定 `theta`、动作集合、`footprint` 格式、`goal` 是否含朝向 | 0.5 天 |
| 已完成 | 2 | 重构状态表示 | [`gridworld.py`](cactus/env/gridworld.py) | `current_positions/goal_positions` 从 2D 点扩成 pose；新增 `footprint` 变换函数 | 1 天 |
| 已完成 | 3 | 重写转移与碰撞 | [`gridworld.py`](cactus/env/gridworld.py), [`collision_gridworld.py`](cactus/env/collision_gridworld.py) | 支持旋转动作、`footprint` 占用、多格碰撞、边交换判定 | 2 天 |
| 已完成（50 epoch smoke） | 4 | 重写 reset / 目标采样 / curriculum 半径 | [`gridworld.py`](cactus/env/gridworld.py), [`curriculum.py`](cactus/curriculum.py) | 生成不重叠初始 pose；`init_goal_radius` 改为基于 anchor 或 swept area 的采样 | 1 天 |
| 已完成（10 epoch smoke） | 5 | 重写观测编码 | [`mapf_gridworld.py`](cactus/env/mapf_gridworld.py) | 观测里加入朝向、`footprint` 占用、可旋转性/可前进性信息 | 1.5 天 |
| 已完成（10 epoch smoke） | 6 | 调整动作 mask 与网络输入输出 | [`a2c_controller.py`](cactus/controller/a2c_controller.py), [`constants.py`](cactus/constants.py) | 新动作空间、非法动作 mask、可能的 observation channel 变更 | 1 天 |
| 已完成（50 epoch smoke） | 7 | 跑通训练与评估 | [`run_training.py`](run_training.py), [`eval.py`](eval.py) | 至少 PPO+QMIX 跑通一个小规模实验 | 1 天 |
| 未完成 | 8 | 单元测试与回归测试 | 新增 `tests`，覆盖环境逻辑 | 测试碰撞、旋转、goal 达成、非法动作、观测一致性 | 1.5 天 |

Notes:

- Phase 4 reset/curriculum refactor is now implemented and smoke-validated.
- Phase 7 currently means a small PPO+QMIX smoke validation completed under the `rl` conda environment.
- The current default training configuration in [`run_training.py`](run_training.py) uses `2` agents and procedurally generated `10x10` maps with densities `0`, `0.1`, `0.2`, and `0.3` to keep iteration time manageable.
- The current default footprint is still single-cell: `DEFAULT_AGENT_FOOTPRINT = ((0, 0),)`.
- Phase 5 observation refactor is now implemented and smoke-validated under the `rl` conda environment with a 10 epoch PPO+QMIX CACTUS run.
- Phase 6 action-mask and network I/O wiring is now implemented and smoke-validated under the `rl` conda environment with a 10 epoch PPO+QMIX CACTUS run.

## Design Notes

- Multi-cell agent refactor spec: [`docs/multi_cell_agent_modeling_spec.md`](docs/multi_cell_agent_modeling_spec.md)

## Prerequisites

Run these commands:

```bash
cd instances
mkdir primal_test_envs
```

### Training maps

Training maps are generated for each training run in `run_training.py` using `cactus.env.env_generator`.

Current defaults:

- map size: `10x10`
- densities: `0`, `0.1`, `0.2`, `0.3`
- number of agents: `2`

### Test maps

Go to the Google Drive referenced by the PRIMAL GitHub repository. Download the archive with all PRIMAL test maps [2] and unpack it in `instances/primal_test_envs`.

## Running the Code

### Training

Run training of all MARL algorithms in the paper with:

```bash
python run_training.py
```

The command creates a folder under `output/` with named result folders per MARL algorithm.

### Test

Run evaluation with (`filename` specifies the result folder with `actor.pth`):

```bash
python eval.py <filename> <map_size> <density>
```

The completion rates are printed on the command line and can be redirected into a text or JSON file for post-processing.

## References

[1] T. Phan et al., [*"Confidence-Based Curriculum Learning for Multi-Agent Path Finding"*](https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p1558.pdf), AAMAS 2024

[2] G. Sartoretti et al., *"PRIMAL: Pathfinding via Reinforcement and Imitation Multi-Agent Learning"*, RA-L 2019
