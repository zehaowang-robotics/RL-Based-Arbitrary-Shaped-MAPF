from cactus.env.environment import Environment
from cactus.utils import assertContains, assertEquals, get_param_or_default
from cactus.constants import *
from cactus.rendering.gridworld_viewer import render
import torch
import random
import heapq

"""
 Represents a 2D grid world
"""
class GridWorld(Environment):

    def __init__(self, params) -> None:
        params[ENV_NR_ACTIONS] = NR_ORIENTED_GRID_ACTIONS
        assertContains(params, ENV_OBSTACLES)
        super(GridWorld, self).__init__(params)
        self.makespan_mode = get_param_or_default(params, ENV_MAKESPAN_MODE, False)
        self.obstacle_map = self.as_bool_tensor(params[ENV_OBSTACLES])
        self.rows = self.obstacle_map.size(0)
        self.columns = self.obstacle_map.size(1)
        self.grid_operations = self.int_zeros((NR_GRID_ACTIONS, ENV_2D))
        self.grid_operations[WAIT]  = self.as_int_tensor([ 0,  0])
        self.grid_operations[NORTH] = self.as_int_tensor([ 0,  1])
        self.grid_operations[SOUTH] = self.as_int_tensor([ 0, -1])
        self.grid_operations[WEST]  = self.as_int_tensor([-1,  0])
        self.grid_operations[EAST]  = self.as_int_tensor([ 1,  0])
        self.delta_to_actions = {
            ( 0, 0) : WAIT,
            ( 0, 1) : NORTH,
            ( 0,-1) : SOUTH,
            (-1, 0) : WEST,
            ( 1, 0) : EAST
        }
        self.nr_orientations = get_param_or_default(params, ENV_NR_ORIENTATIONS, DEFAULT_NR_ORIENTATIONS)
        assert self.nr_orientations == DEFAULT_NR_ORIENTATIONS, "Only four discrete orientations are supported"
        self.agent_footprint = self.normalize_footprint(get_param_or_default(params, ENV_AGENT_FOOTPRINT, DEFAULT_AGENT_FOOTPRINT))
        self.goal_orientation_required = get_param_or_default(params, ENV_GOAL_ORIENTATION_REQUIRED, False)
        self.orientation_deltas = self.as_int_tensor([
            [ 1,  0],
            [ 0,  1],
            [-1,  0],
            [ 0, -1],
        ])
        self.occupiable_locations = self.get_occupiable_locations()
        self.shortest_distance_maps = {}
        self.current_positions = self.int_zeros([self.nr_agents, ENV_POSE_DIM])
        self.goal_positions = self.int_zeros([self.nr_agents, ENV_POSE_DIM])
        raw_goal_positions = get_param_or_default(params, ENV_GOAL_POSES, get_param_or_default(params, ENV_GOAL_POSITIONS, None))
        raw_start_positions = get_param_or_default(params, ENV_START_POSES, get_param_or_default(params, ENV_START_POSITIONS, None))
        self.init_goal_poses = self.as_pose_batch(raw_goal_positions)
        self.init_start_poses = self.as_pose_batch(raw_start_positions)
        # Keep the legacy attribute names alive while their payload becomes pose-shaped.
        self.init_goal_positions = self.init_goal_poses
        self.init_start_positions = self.init_start_poses
        self.collision_weight = get_param_or_default(params, ENV_COLLISION_WEIGHT, 0)
        self.time_penalty = get_param_or_default(params, ENV_TIME_PENALTY, -1.0)
        self.init_goal_radius = get_param_or_default(params, ENV_INIT_GOAL_RADIUS, None)
        self.completion_reward = get_param_or_default(params, ENV_COMPLETION_REWARD, 1.0)
        self.use_primal_reward = get_param_or_default(params, ENV_USE_PRIMAL_REWARD, False)
        if self.use_primal_reward:
            self.collision_weight = 2
            self.time_penalty = -0.3
            self.completion_reward = 20.0
        self.occupied_goal_positions = -self.int_ones([self.rows, self.columns])
        self.shortest_distance_map = -self.int_ones([self.nr_agents, self.rows, self.columns])
        self.current_position_map = -self.int_ones_like(self.obstacle_map)
        self.next_position_map = -self.int_ones_like(self.obstacle_map)
        self.viewer = None

    def normalize_footprint(self, footprint):
        normalized = []
        seen_offsets = set()
        for cell in footprint:
            assertEquals(ENV_2D, len(cell))
            offset = (int(cell[0]), int(cell[1]))
            if offset not in seen_offsets:
                normalized.append(offset)
                seen_offsets.add(offset)
        assert (0, 0) in normalized, "Footprint must include the anchor cell (0, 0)"
        return tuple(normalized)

    def as_pose(self, pose, default_theta=THETA_0):
        if torch.is_tensor(pose):
            pose_tensor = pose.to(device=self.device, dtype=INT_TYPE).view(-1).clone()
        else:
            pose_tensor = self.as_int_tensor(pose).view(-1)
        if pose_tensor.numel() == ENV_2D:
            full_pose = self.int_zeros(ENV_POSE_DIM)
            full_pose[:ENV_2D] = pose_tensor
            full_pose[ENV_2D] = default_theta
            pose_tensor = full_pose
        else:
            assertEquals(ENV_POSE_DIM, pose_tensor.numel())
        pose_tensor[ENV_2D] = torch.remainder(pose_tensor[ENV_2D], self.nr_orientations)
        return pose_tensor

    def as_pose_batch(self, poses, default_theta=THETA_0):
        if poses is None:
            return None
        if torch.is_tensor(poses):
            pose_tensor = poses.to(device=self.device, dtype=INT_TYPE).clone()
        else:
            pose_tensor = self.as_int_tensor(poses)
        assertEquals(2, pose_tensor.dim())
        assertEquals(self.nr_agents, pose_tensor.size(0))
        if pose_tensor.size(1) == ENV_2D:
            full_poses = self.int_zeros([self.nr_agents, ENV_POSE_DIM])
            full_poses[:, :ENV_2D] = pose_tensor
            full_poses[:, ENV_2D] = default_theta
            pose_tensor = full_poses
        else:
            assertEquals(ENV_POSE_DIM, pose_tensor.size(1))
        pose_tensor[:, ENV_2D] = torch.remainder(pose_tensor[:, ENV_2D], self.nr_orientations)
        return pose_tensor

    def anchor_positions(self, poses):
        return poses[..., :ENV_2D]

    def pose_orientations(self, poses):
        return poses[..., ENV_2D]

    def translate_poses(self, poses, deltas):
        translated = poses.clone()
        translated[..., :ENV_2D] += deltas
        return translated

    def rotate_offset(self, offset, theta):
        dx, dy = int(offset[0]), int(offset[1])
        theta = int(theta) % self.nr_orientations
        if theta == THETA_0:
            return (dx, dy)
        if theta == THETA_90:
            return (-dy, dx)
        if theta == THETA_180:
            return (-dx, -dy)
        if theta == THETA_270:
            return (dy, -dx)
        raise ValueError(f"Unsupported orientation index: {theta}")

    def transformed_footprint(self, theta, footprint=None):
        if footprint is None:
            footprint = self.agent_footprint
        transformed = [self.rotate_offset(offset, theta) for offset in footprint]
        return self.as_int_tensor(transformed).view(-1, ENV_2D)

    def occupied_cells_from_pose(self, pose, footprint=None):
        pose_tensor = self.as_pose(pose)
        footprint_offsets = self.transformed_footprint(pose_tensor[ENV_2D].item(), footprint)
        return footprint_offsets + pose_tensor[:ENV_2D].view(1, ENV_2D)

    def occupied_cells_from_poses(self, poses, footprint=None):
        pose_batch = self.as_pose_batch(poses)
        occupied_cells = [self.occupied_cells_from_pose(pose, footprint) for pose in pose_batch]
        return self.stack(occupied_cells)

    def populate_position_map(self, position_map, poses):
        position_map[:] = -1
        occupied_cells = self.occupied_cells_from_poses(poses)
        for agent_id, cells in enumerate(occupied_cells):
            position_map[cells[:,0], cells[:,1]] = agent_id
        return occupied_cells

    def has_init_configuration(self):
        return self.init_goal_poses is not None and self.init_start_poses is not None

    def render(self):
        self.viewer = render(self, self.viewer)

    def get_occupiable_locations(self):
        return [(r,c) for r in range(self.rows) for c in range(self.columns) if not self.obstacle_map[r][c]]

    def heading_deltas(self, orientations):
        return self.orientation_deltas[orientations.view(-1)].view(-1, ENV_2D)

    def transition_poses(self, joint_action):
        next_positions = self.current_positions.clone()
        headings = self.heading_deltas(self.pose_orientations(self.current_positions))
        forward_mask = joint_action == FORWARD
        backward_mask = joint_action == BACKWARD
        rotate_left_mask = joint_action == ROTATE_LEFT
        rotate_right_mask = joint_action == ROTATE_RIGHT
        next_positions[forward_mask, :ENV_2D] += headings[forward_mask]
        next_positions[backward_mask, :ENV_2D] -= headings[backward_mask]
        next_positions[rotate_left_mask, ENV_2D] = torch.remainder(next_positions[rotate_left_mask, ENV_2D] + 1, self.nr_orientations)
        next_positions[rotate_right_mask, ENV_2D] = torch.remainder(next_positions[rotate_right_mask, ENV_2D] - 1, self.nr_orientations)
        return next_positions

    def step(self, joint_action):
        self.time_step += 1
        if self.time_step >= self.time_limit or self.is_done().all():
            assert self.is_done().all()
            terminated = self.is_terminated()
            return self.joint_observation(), self.float_zeros(self.nr_agents), terminated, self.is_truncated(), {
                ENV_VERTEX_COLLISIONS: self.bool_zeros(self.nr_agents),
                ENV_EDGE_COLLISIONS: self.bool_zeros(self.nr_agents),
                ENV_COMPLETION_RATE: terminated.to(FLOAT_TYPE).sum()/self.nr_agents
            }
        joint_action = joint_action.to(INT_TYPE).view(-1)
        is_done_before = self.is_terminated()
        not_done = torch.logical_not(is_done_before)
        new_positions = self.transition_poses(joint_action)
        self.current_positions, collisions = self.move_to(new_positions)
        self.populate_position_map(self.current_position_map, self.current_positions)
        is_done_now = self.is_terminated()
        if self.makespan_mode:
            reward = self.float_ones(self.nr_agents)
            if not self.is_truncated().all():
                reward *= -1
        else:
            reward = torch.where(torch.logical_and(is_done_now, not_done), 1.0, self.time_penalty).to(FLOAT_TYPE)
            was_done = torch.logical_and(is_done_now, is_done_before)
            reward = torch.where(torch.logical_and(is_done_now, is_done_before), self.float_zeros_like(reward), reward)
            if self.use_primal_reward:
                all_wait = (joint_action == WAIT).all()
                was_not_done = torch.logical_not(was_done)
                reward = torch.where(torch.logical_and(was_not_done, all_wait), -0.5, reward)
        vertex_collisions, edge_collisions = collisions
        if self.collision_weight is not None:
            vertex_collisions = vertex_collisions.to(INT_TYPE)
            edge_collisions = edge_collisions.to(INT_TYPE)/2
            reward -= self.collision_weight*(vertex_collisions + edge_collisions).to(FLOAT_TYPE)
        terminated = self.is_terminated()
        if terminated.all():
            reward += self.completion_reward
        self.undiscounted_returns += reward
        self.discounted_returns += (self.gamma**(self.time_step-1))*reward
        return self.joint_observation(), reward, terminated, self.is_truncated(),\
            {
                ENV_VERTEX_COLLISIONS: vertex_collisions,
                ENV_EDGE_COLLISIONS: edge_collisions,
                ENV_COMPLETION_RATE: terminated.to(FLOAT_TYPE).sum()/self.nr_agents
            }

    def is_terminated(self):
        current_anchor_positions = self.anchor_positions(self.current_positions)
        goal_anchor_positions = self.anchor_positions(self.goal_positions)
        x_equal = current_anchor_positions[:,0] == goal_anchor_positions[:,0]
        y_equal = current_anchor_positions[:,1] == goal_anchor_positions[:,1]
        anchor_match = torch.logical_and(x_equal, y_equal)
        if not self.goal_orientation_required:
            return anchor_match
        return torch.logical_and(anchor_match, self.pose_orientations(self.current_positions) == self.pose_orientations(self.goal_positions))

    def position_in_bounds(self, pos):
        occupied_cells = self.occupied_cells_from_poses(pos)
        x = occupied_cells[:,:,0].reshape(-1)
        y = occupied_cells[:,:,1].reshape(-1)
        valid_cells = self.xy_position_in_bounds(x, y).view(self.nr_agents, -1)
        return valid_cells.all(-1)

    def move_condition(self, new_positions):
        in_bounds = self.position_in_bounds(new_positions)
        return in_bounds.unsqueeze(1).expand_as(new_positions), (self.bool_zeros(self.nr_agents), self.bool_zeros(self.nr_agents))

    def move_to(self, new_positions):
        new_positions_changed = True
        vertex_collisions = self.bool_zeros(self.nr_agents)
        edge_collisions = self.bool_zeros(self.nr_agents)
        while new_positions_changed:
            condition, new_collisions = self.move_condition(new_positions)
            new_positions_1 = torch.where(condition, new_positions, self.current_positions)
            vertex_collisions = torch.logical_or(vertex_collisions, new_collisions[0])
            edge_collisions = torch.logical_or(edge_collisions, new_collisions[1])
            new_positions_changed = (new_positions_1 != new_positions).any()
            new_positions = new_positions_1
        return torch.where(condition, new_positions, self.current_positions), (vertex_collisions, edge_collisions)

    def set_init_goal_radius(self, radius):
        self.init_goal_radius = radius

    def increment_init_goal_radius(self):
        self.init_goal_radius += 1

    def decrement_init_goal_radius(self):
        self.init_goal_radius -= 1

    def xy_position_in_bounds(self, x, y):
        nonnegative = torch.logical_and(x >= 0, y >= 0)
        in_bounds = torch.logical_and(x < self.rows, y < self.columns)
        in_bounds = torch.logical_and(in_bounds, nonnegative)
        x_clamped = x.clamp(0, self.rows-1)
        y_clamped = y.clamp(0, self.columns-1)
        no_obstacle = torch.logical_not(self.obstacle_map[x_clamped, y_clamped])
        return torch.logical_and(in_bounds, no_obstacle)

    def get_neighbor_positions(self, position, delta):
        assert delta > 0
        x, y = position[0], position[1]
        neighbors = []
        for dx in range(-delta, delta+1):
            for dy in range(-delta, delta+1):
                new_pos = (x+dx, y+dy)
                x1, y1 = new_pos
                no_overlap = (dx, dy) != (0,0)
                in_bounds = x1 >= 0 and y1 >= 0 and x1 < self.rows and y1 < self.columns
                if in_bounds:
                    no_obstacle = not self.obstacle_map[x1][y1]
                    not_occupied = self.occupied_goal_positions[x1][y1] < 0
                else:
                    no_obstacle = False
                    not_occupied = False
                if no_overlap and in_bounds and no_obstacle and not_occupied:
                    neighbors.append(new_pos)
        return neighbors

    def set_start_positions(self, positions):
        self.current_positions[:] = self.as_pose_batch(positions)
        self.populate_position_map(self.current_position_map, self.current_positions)

    def set_goal_positions(self, positions):
        poses = self.as_pose_batch(positions)
        self.occupied_goal_positions[:] = -1
        self.goal_positions[:] = poses
        goal_anchor_positions = self.anchor_positions(poses)
        for a in range(self.nr_agents):
            x, y = goal_anchor_positions[a,0], goal_anchor_positions[a,1]
            self.occupied_goal_positions[x][y] = a

    def reset(self):
        super(GridWorld, self).reset()
        self.occupied_goal_positions[:] = -1
        self.current_positions[:] = 0
        self.goal_positions[:] = 0
        if self.has_init_configuration():
            self.set_start_positions(self.init_start_poses)
            self.set_goal_positions(self.init_goal_poses)
            return self.joint_observation()
        random.shuffle(self.occupiable_locations)
        nr_samples = 2*self.nr_agents
        sampled_locations = random.sample(self.occupiable_locations, k=nr_samples)
        for a in range(self.nr_agents):
            index = a*2
            x, y = sampled_locations[index][0], sampled_locations[index][1]
            self.current_positions[a,0] = x
            self.current_positions[a,1] = y
            self.current_positions[a,ENV_2D] = THETA_0
            if self.init_goal_radius is not None and self.init_goal_radius < max(self.rows, self.columns):
                goal_candidates = self.get_neighbor_positions((x, y), self.init_goal_radius)
                sampled_location = random.choice(goal_candidates)
                self.goal_positions[a,0] = sampled_location[0]
                self.goal_positions[a,1] = sampled_location[1]
                self.goal_positions[a,ENV_2D] = THETA_0
                self.occupied_goal_positions[sampled_location[0]][sampled_location[1]] = a
            else:
                self.goal_positions[a,0] = sampled_locations[index+1][0]
                self.goal_positions[a,1] = sampled_locations[index+1][1]
                self.goal_positions[a,ENV_2D] = THETA_0
                self.occupied_goal_positions[self.goal_positions[a,0]][self.goal_positions[a,1]] = a
        self.populate_position_map(self.current_position_map, self.current_positions)
        assertEquals(self.nr_agents, len(self.occupied_goal_positions[self.occupied_goal_positions >= 0].view(-1)))
        return self.joint_observation()

    def get_adjacent_neighbors(self, pos):
        x, y = pos
        neighbors = []
        if x > 0 and not self.obstacle_map[x-1][y]:
            neighbors.append((x-1, y))
        if x < self.rows-1 and not self.obstacle_map[x+1][y]:
            neighbors.append((x+1, y))
        if y > 0 and not self.obstacle_map[x][y-1]:
            neighbors.append((x, y-1))
        if y < self.columns-1 and not self.obstacle_map[x][y+1]:
            neighbors.append((x, y+1))
        return neighbors

    def compute_shortest_distances(self):
        goal_anchor_positions = self.anchor_positions(self.goal_positions)
        for i in range(self.nr_agents):
            goal_position = (goal_anchor_positions[i,0].item(), goal_anchor_positions[i,1].item())
            self.shortest_distance_map[i,:,:] = self.shortest_distance_maps[goal_position]

    def compute_shortest_distances_for(self, map_tensor, goal_position):
        x, y = goal_position
        pos = (x, y)
        queue = [(0, pos)]
        while len(queue) > 0:
            current_distance, current_vertex = heapq.heappop(queue)
            x0, y0 = current_vertex
            stored_distance = map_tensor[x0][y0]
            if stored_distance < 0 or current_distance <= stored_distance:
                x0, y0 = current_vertex
                map_tensor[x0][y0] = current_distance
                for neighbor in self.get_adjacent_neighbors(current_vertex):
                    distance = current_distance + 1
                    x1, y1 = neighbor
                    stored_distance = map_tensor[x1][y1]
                    if stored_distance < 0 or distance < stored_distance:
                        x1, y1 = neighbor
                        map_tensor[x1][y1] = distance
                        heapq.heappush(queue, (distance, neighbor))
        return map_tensor

    def print(self):
        map_tensor = self.int_zeros_like(self.obstacle_map) - self.obstacle_map.to(INT_TYPE)
        current_anchor_positions = self.anchor_positions(self.current_positions)
        goal_anchor_positions = self.anchor_positions(self.goal_positions)
        x = current_anchor_positions[:,0]
        y = current_anchor_positions[:,1]
        map_tensor[x,y] = 1
        x = goal_anchor_positions[:,0]
        y = goal_anchor_positions[:,1]
        map_tensor[x,y] = 2
        for x in range(self.rows):
            line = ""
            for y in range(self.columns):
                if map_tensor[x][y] < 0:
                    line += "# "
                elif map_tensor[x][y] == 1:
                    line += "O "
                elif map_tensor[x][y] == 2:
                    line += "X "
                else:
                    line += ". "
            print(line)