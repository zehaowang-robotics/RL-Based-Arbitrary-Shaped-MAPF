# RL-Based Arbitrary-Shaped Multi-Agent Path Finding

This repository is based on `rl4mapf` and is being extended toward arbitrary-shaped, multi-cell MAPF agents with rotation.

## Featured Algorithm

- CACTUS: Confidence-based Auto-Curriculum for Team Update Stability [1]

## Refactor Progress

| Status | Phase | Goal | Main Files | Deliverable | Estimate |
| --- | --- | --- | --- | --- | --- |
| Completed | 1 | Freeze the modeling specification | New configuration keys and constants cleanup | Define `theta`, the action set, the `footprint` format, and whether goals include orientation | 0.5 days |
| Completed | 2 | Refactor the state representation | [`gridworld.py`](cactus/env/gridworld.py) | Extend `current_positions` and `goal_positions` from 2D points to poses; add footprint transformation helpers | 1 day |
| Completed | 3 | Rewrite transitions and collisions | [`gridworld.py`](cactus/env/gridworld.py), [`collision_gridworld.py`](cactus/env/collision_gridworld.py) | Support rotation actions, footprint occupancy, multi-cell collisions, swept rotation collision checks, and generalized edge swaps | 2 days |
| Completed (50-epoch smoke) | 4 | Rewrite reset, goal sampling, and curriculum radius | [`gridworld.py`](cactus/env/gridworld.py), [`curriculum.py`](cactus/curriculum.py) | Generate non-overlapping initial poses; support `init_goal_radius` based on anchor distance or swept-area distance | 1 day |
| Completed (10-epoch smoke) | 5 | Rewrite observation encoding | [`mapf_gridworld.py`](cactus/env/mapf_gridworld.py) | Add orientation, footprint occupancy, and action-feasibility information to observations | 1.5 days |
| Completed (10-epoch smoke) | 6 | Adjust action masks and network I/O | [`a2c_controller.py`](cactus/controller/a2c_controller.py), [`constants.py`](cactus/constants.py) | Wire the oriented action space, invalid-action masks, and observation-channel changes into the controller | 1 day |
| Completed (50-epoch smoke) | 7 | Run training and evaluation | [`run_training.py`](run_training.py), [`eval.py`](eval.py) | Run at least one small-scale PPO+QMIX experiment successfully | 1 day |
| Not started | 8 | Add unit and regression tests | New `tests` coverage for environment logic | Test collisions, rotations, goal completion, invalid actions, and observation consistency | 1.5 days |

Notes:

- Phase 4 reset and curriculum refactor is implemented and smoke-validated.
- Phase 5 observation refactor is implemented and smoke-validated under the `rl` conda environment with a 10-epoch PPO+QMIX CACTUS run.
- Phase 6 action-mask and network I/O wiring is implemented and smoke-validated under the `rl` conda environment with a 10-epoch PPO+QMIX CACTUS run.
- Phase 7 currently means a small PPO+QMIX smoke validation completed under the `rl` conda environment.
- The current default training configuration in [`run_training.py`](run_training.py) uses `1` agent.
- Training maps are procedurally generated at sizes `10x10`, `20x20`, `40x40`, and `80x80`, with obstacle densities `0`, `0.1`, `0.2`, and `0.3`.
- The default training entry point currently runs PPO+QMIX with the CACTUS curriculum only; the other algorithm runs are left commented out in [`run_training.py`](run_training.py).
- The current default footprint is L-shaped: `DEFAULT_AGENT_FOOTPRINT = ((0, 0), (0, 1), (1, 0))`, with the anchor/state point at `(0, 0)`.
- The oriented action space includes `WAIT`, `FORWARD`, `BACKWARD`, `ROTATE_LEFT`, `ROTATE_RIGHT`, `STRAFE_LEFT`, and `STRAFE_RIGHT`.
- Rotation collision checks use the grid cells swept by the footprint during the quarter-turn, not only the cells occupied before and after rotation.

## Design Notes

- Multi-cell agent refactor spec: [`docs/multi_cell_agent_modeling_spec.md`](docs/multi_cell_agent_modeling_spec.md)

## Prerequisites

Run these commands:

```bash
cd instances
mkdir primal_test_envs
```

### Training Maps

Training maps are generated for each training run in `run_training.py` using `cactus.env.env_generator`.

Current defaults:

- map sizes: `10x10`, `20x20`, `40x40`, `80x80`
- densities: `0`, `0.1`, `0.2`, `0.3`
- number of agents: `1`

### Test Maps

Go to the Google Drive referenced by the PRIMAL GitHub repository. Download the archive with all PRIMAL test maps [2] and unpack it in `instances/primal_test_envs`.

## Running the Code

### Training

Run the default training configuration with:

```bash
python run_training.py
```

The command creates a folder under `output/` with named result folders per MARL algorithm and curriculum setting.

### Test

Run evaluation with `filename` specifying the result folder that contains `actor.pth`:

```bash
python eval.py <filename> <map_size> <density>
```

The completion rates are printed on the command line and can be redirected into a text or JSON file for post-processing.

## References

[1] T. Phan et al., [*"Confidence-Based Curriculum Learning for Multi-Agent Path Finding"*](https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p1558.pdf), AAMAS 2024

[2] G. Sartoretti et al., *"PRIMAL: Pathfinding via Reinforcement and Imitation Multi-Agent Learning"*, RA-L 2019
