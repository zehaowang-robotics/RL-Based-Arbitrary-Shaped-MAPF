import argparse
import json
import random
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import cactus.algorithms as algorithms
import cactus.env.env_generator as env_generator
from cactus.constants import *


DEFAULT_MODEL_DIR = Path("output") / "1-agents_CACTUS_PPO_QMIX_2026-04-13-19-52-42"
COLORS = [(218, 62, 72), (43, 116, 255), (237, 145, 33), (176, 81, 219), (25, 148, 117), (98, 94, 177)]


def torch_load_state(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def infer_model_params(model_dir, obs_size):
    actor_state = torch_load_state(model_dir / ACTOR_NET_FILENAME)
    input_shape = int(actor_state["fc1.0.weight"].shape[1])
    hidden_layer_dim = int(actor_state["fc1.0.weight"].shape[0])
    nr_actions = int(actor_state["output.weight"].shape[0])
    cells = obs_size * obs_size
    if input_shape % cells != 0:
        raise ValueError(f"Cannot infer observation channels: input shape {input_shape} is not divisible by {cells}")
    if nr_actions == NR_GRID_ACTIONS:
        action_space = ACTION_SPACE_CARDINAL
    elif nr_actions == NR_ORIENTED_GRID_ACTIONS:
        action_space = ACTION_SPACE_ORIENTED
    else:
        raise ValueError(f"Unsupported action count in actor output: {nr_actions}")

    nr_agents = 1
    mixing_hidden_size = 64
    mixer_path = model_dir / MIXER_NET_FILENAME
    if mixer_path.exists():
        mixer_state = torch_load_state(mixer_path)
        mixer_input_shape = int(mixer_state["hyper_w_1.0.weight"].shape[1])
        if mixer_input_shape % input_shape != 0:
            raise ValueError("Cannot infer number of agents from mixer input shape")
        nr_agents = mixer_input_shape // input_shape
        mixing_hidden_size = int(mixer_state["hyper_w_1.0.weight"].shape[0])
    return {
        "input_shape": input_shape,
        "observation_channels": input_shape // cells,
        "hidden_layer_dim": hidden_layer_dim,
        "nr_actions": nr_actions,
        "action_space": action_space,
        "nr_agents": nr_agents,
        "mixing_hidden_size": mixing_hidden_size,
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_params(inferred, args):
    params = {}
    params[ENV_OBSERVATION_SIZE] = args.obs_size
    params[ENV_ACTION_SPACE] = inferred["action_space"]
    params[ENV_NR_AGENTS] = args.nr_agents if args.nr_agents is not None else inferred["nr_agents"]
    params[SAMPLE_NR_AGENTS] = params[ENV_NR_AGENTS]
    params[HIDDEN_LAYER_DIM] = inferred["hidden_layer_dim"]
    params[NUMBER_OF_EPOCHS] = 1
    params[EPISODES_PER_EPOCH] = 1
    params[EPOCH_LOG_INTERVAL] = 1
    params[ENV_TIME_LIMIT] = args.time_limit
    params[TEST_INIT_GOAL_RADIUS] = args.goal_radius
    params[ENV_GAMMA] = 1
    params[RENDER_MODE] = False
    params[ENV_MAKESPAN_MODE] = False
    params[GRAD_NORM_CLIP] = 10
    params[VDN_MODE] = False
    params[REWARD_SHARING] = False
    params[MIXING_HIDDEN_SIZE] = inferred["mixing_hidden_size"]
    params[ALGORITHM_NAME] = ALGORITHM_PPO_QMIX
    if args.goal_radius is not None:
        params[ENV_INIT_GOAL_RADIUS] = args.goal_radius
    return params


def make_env_and_controller(model_dir, inferred, args, seed):
    set_seed(seed)
    params = make_params(inferred, args)
    env, params = env_generator.generate_mapf_gridworld(params[ENV_NR_AGENTS], args.size, args.density, params)
    controller = algorithms.make(params)
    controller.load_model_weights(str(model_dir))
    return env, controller


def select_action(controller, observations, greedy):
    with torch.no_grad():
        joint_observation = observations.view(1, controller.nr_agents, -1)
        action_mask = controller.calculate_action_masks(joint_observation)
        logits = controller.policy_network(joint_observation)
        probs = F.softmax(logits + action_mask, dim=-1)
        if greedy:
            action = probs.argmax(dim=-1)
        else:
            action = torch.distributions.Categorical(probs).sample()
    return action.view(controller.nr_agents), probs.view(controller.nr_agents, controller.nr_actions)


def tensor_to_list(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def snapshot(env, action=None, reward=None, info=None, probs=None):
    terminated = env.is_terminated()
    return {
        "time_step": int(env.time_step),
        "current_positions": env.current_positions.detach().cpu().clone(),
        "goal_positions": env.goal_positions.detach().cpu().clone(),
        "terminated": terminated.detach().cpu().clone(),
        "completion_rate": float(terminated.to(FLOAT_TYPE).mean().item()),
        "action": tensor_to_list(action),
        "reward": tensor_to_list(reward),
        "info": {key: tensor_to_list(value) for key, value in (info or {}).items()},
        "probs": tensor_to_list(probs),
    }


def run_episode(env, controller, max_steps, greedy):
    observations = env.reset()
    snapshots = [snapshot(env)]
    last_reward = None
    last_info = None
    while not bool(env.is_done_all().item()) and env.time_step < max_steps:
        action, probs = select_action(controller, observations, greedy)
        observations, reward, _terminated, _truncated, info = env.step(action)
        last_reward = reward
        last_info = info
        snapshots.append(snapshot(env, action=action, reward=reward, info=info, probs=probs))
    return {
        "snapshots": snapshots,
        "success": bool(env.is_terminated().all().item()),
        "completion_rate": float(env.is_terminated().to(FLOAT_TYPE).mean().item()),
        "steps": int(env.time_step),
        "returns": tensor_to_list(env.undiscounted_returns),
        "last_reward": tensor_to_list(last_reward),
        "last_info": {key: tensor_to_list(value) for key, value in (last_info or {}).items()},
    }


def cell_rect(row, col, cell_size):
    left = col * cell_size
    top = row * cell_size
    return [left, top, left + cell_size, top + cell_size]


def cell_center(row, col, cell_size):
    return (col * cell_size + cell_size // 2, row * cell_size + cell_size // 2)


def draw_base(env, cell_size):
    rows = int(env.rows)
    cols = int(env.columns)
    image = Image.new("RGB", (cols * cell_size, rows * cell_size), (248, 248, 248))
    draw = ImageDraw.Draw(image)
    obstacles = env.obstacle_map.detach().cpu().numpy()
    for row in range(rows):
        for col in range(cols):
            rect = cell_rect(row, col, cell_size)
            draw.rectangle(rect, fill=(42, 42, 42) if obstacles[row, col] else (248, 248, 248))
            draw.rectangle(rect, outline=(215, 215, 215))
    return image


def occupied_cells(env, poses):
    return env.occupied_cells_from_poses(poses.to(env.device)).detach().cpu().numpy()


def draw_pose_set(env, draw, poses, cell_size, fill_agents, outline_goals):
    cells_by_agent = occupied_cells(env, poses)
    anchors = env.anchor_positions(poses.to(env.device)).detach().cpu().numpy()
    headings = env.heading_deltas(env.pose_orientations(poses.to(env.device))).detach().cpu().numpy()
    for agent_id, cells in enumerate(cells_by_agent):
        color = COLORS[agent_id % len(COLORS)]
        for row, col in cells:
            rect = cell_rect(int(row), int(col), cell_size)
            if fill_agents:
                draw.rectangle(rect, fill=color, outline=(20, 20, 20), width=1)
            if outline_goals:
                inset = max(2, cell_size // 8)
                goal_rect = [rect[0] + inset, rect[1] + inset, rect[2] - inset, rect[3] - inset]
                draw.rectangle(goal_rect, outline=color, width=max(2, cell_size // 8))
        row = int(anchors[agent_id][0])
        col = int(anchors[agent_id][1])
        center = cell_center(row, col, cell_size)
        if fill_agents:
            radius = max(3, cell_size // 5)
            draw.ellipse([center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius], fill=(255, 255, 255), outline=(20, 20, 20))
            end = (int(center[0] + headings[agent_id][1] * cell_size * 0.35), int(center[1] + headings[agent_id][0] * cell_size * 0.35))
            draw.line([center, end], fill=(20, 20, 20), width=max(2, cell_size // 12))


def draw_snapshot(env, snap, cell_size):
    image = draw_base(env, cell_size)
    draw = ImageDraw.Draw(image)
    draw_pose_set(env, draw, snap["goal_positions"], cell_size, fill_agents=False, outline_goals=True)
    draw_pose_set(env, draw, snap["current_positions"], cell_size, fill_agents=True, outline_goals=False)
    return image


def draw_trajectory(env, snapshots, cell_size):
    image = draw_base(env, cell_size)
    draw = ImageDraw.Draw(image)
    first = snapshots[0]
    last = snapshots[-1]
    draw_pose_set(env, draw, first["goal_positions"], cell_size, fill_agents=False, outline_goals=True)
    nr_agents = int(first["current_positions"].shape[0])
    for agent_id in range(nr_agents):
        points = []
        for snap in snapshots:
            anchors = env.anchor_positions(snap["current_positions"].to(env.device)).detach().cpu().numpy()
            row = int(anchors[agent_id][0])
            col = int(anchors[agent_id][1])
            points.append(cell_center(row, col, cell_size))
        color = COLORS[agent_id % len(COLORS)]
        if len(points) > 1:
            draw.line(points, fill=color, width=max(2, cell_size // 8))
        stride = max(1, len(points) // 24)
        for point in points[::stride]:
            radius = max(2, cell_size // 10)
            draw.ellipse([point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius], fill=color)
    draw_pose_set(env, draw, first["current_positions"], cell_size, fill_agents=True, outline_goals=False)
    draw_pose_set(env, draw, last["current_positions"], cell_size, fill_agents=True, outline_goals=False)
    return image


def serializable_summary(result, seed, inferred, args):
    return {
        "seed": seed,
        "success": result["success"],
        "completion_rate": result["completion_rate"],
        "steps": result["steps"],
        "returns": result["returns"],
        "model": inferred,
        "map_size": args.size,
        "density": args.density,
        "goal_radius": args.goal_radius,
        "greedy": args.greedy,
    }


def save_demo(env, result, output_prefix, fps, cell_size):
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    frames = [draw_snapshot(env, snap, cell_size) for snap in result["snapshots"]]
    gif_path = output_prefix.with_suffix(".gif")
    png_path = output_prefix.with_name(output_prefix.name + "_trajectory.png")
    imageio.mimsave(gif_path, [np.asarray(frame) for frame in frames], duration=1.0 / max(1, fps), loop=0)
    draw_trajectory(env, result["snapshots"], cell_size).save(png_path)
    return gif_path, png_path


def main():
    parser = argparse.ArgumentParser(description="Render a trained MAPF agent demo as a GIF and trajectory image.")
    parser.add_argument("model_dir", nargs="?", default=str(DEFAULT_MODEL_DIR), help="Directory containing actor_net.pth and related weights.")
    parser.add_argument("--out-prefix", default=None, help="Output prefix. Defaults to <model_dir>/agent_demo.")
    parser.add_argument("--size", type=int, default=10, help="Generated demo map size.")
    parser.add_argument("--density", type=float, default=0.0, help="Generated obstacle density.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for map/start/goal sampling.")
    parser.add_argument("--search-success", type=int, default=0, help="Try this many consecutive seeds and keep the first successful demo.")
    parser.add_argument("--goal-radius", type=int, default=None, help="Optional start-goal sampling radius.")
    parser.add_argument("--time-limit", type=int, default=256, help="Episode time limit.")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum rendered steps. Defaults to the time limit.")
    parser.add_argument("--obs-size", type=int, default=7, help="Observation window size used by the trained model.")
    parser.add_argument("--nr-agents", type=int, default=None, help="Override inferred number of agents.")
    parser.add_argument("--fps", type=int, default=8, help="GIF frames per second.")
    parser.add_argument("--cell-size", type=int, default=28, help="Rendered cell size in pixels.")
    parser.add_argument("--sample", action="store_true", help="Sample stochastic actions instead of greedy actions.")
    args = parser.parse_args()
    args.greedy = not args.sample

    model_dir = Path(args.model_dir)
    inferred = infer_model_params(model_dir, args.obs_size)
    output_prefix = Path(args.out_prefix) if args.out_prefix is not None else model_dir / "agent_demo"
    max_steps = args.max_steps if args.max_steps is not None else args.time_limit
    attempts = max(1, args.search_success)
    best = None
    for offset in range(attempts):
        seed = args.seed + offset
        env, controller = make_env_and_controller(model_dir, inferred, args, seed)
        result = run_episode(env, controller, max_steps, args.greedy)
        if best is None or (result["completion_rate"], -result["steps"]) > (best[1]["completion_rate"], -best[1]["steps"]):
            best = (env, result, seed)
        if result["success"]:
            best = (env, result, seed)
            break

    env, result, seed = best
    gif_path, png_path = save_demo(env, result, output_prefix, args.fps, args.cell_size)
    summary = serializable_summary(result, seed, inferred, args)
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Model params: agents={inferred['nr_agents']}, actions={inferred['nr_actions']}, action_space={inferred['action_space']}")
    print(f"Selected seed: {seed}")
    print(f"Success/completion/steps: {result['success']}/{result['completion_rate']:.3f}/{result['steps']}")
    print(f"Saved GIF: {gif_path}")
    print(f"Saved trajectory: {png_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
