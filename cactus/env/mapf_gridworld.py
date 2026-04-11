from cactus.env.collision_gridworld import CollisionGridWorld
from cactus.utils import assertContains
from cactus.constants import *
import torch

"""
 Represents a navigation task in a 2D grid world environment with collision detection.
"""
class MAPFGridWorld(CollisionGridWorld):

    def __init__(self, params) -> None:
        assertContains(params, ENV_OBSERVATION_SIZE)
        nr_orientations = params[ENV_NR_ORIENTATIONS] if ENV_NR_ORIENTATIONS in params else DEFAULT_NR_ORIENTATIONS
        legacy_channels = MAPF_LEGACY_OBSERVATION_CHANNELS
        self.nr_channels = legacy_channels + 2 * nr_orientations + 2 + NR_ORIENTED_GRID_ACTIONS
        self.observation_size = params["observation_size"]
        params[ENV_OBSERVATION_DIM] = [self.nr_channels, self.observation_size, self.observation_size]
        super(MAPFGridWorld, self).__init__(params)
        self.self_orientation_channel = legacy_channels
        self.self_footprint_channel = self.self_orientation_channel + self.nr_orientations
        self.other_footprint_channel = self.self_footprint_channel + 1
        self.other_orientation_channel = self.other_footprint_channel + 1
        self.action_feasibility_channel = self.other_orientation_channel + self.nr_orientations
        params[ENV_MAPF_SELF_ORIENTATION_CHANNEL] = self.self_orientation_channel
        params[ENV_MAPF_SELF_FOOTPRINT_CHANNEL] = self.self_footprint_channel
        params[ENV_MAPF_OTHER_FOOTPRINT_CHANNEL] = self.other_footprint_channel
        params[ENV_MAPF_OTHER_ORIENTATION_CHANNEL] = self.other_orientation_channel
        params[ENV_MAPF_ACTION_FEASIBILITY_CHANNEL] = self.action_feasibility_channel
        self.observation_dx, self.observation_dy = self.get_delta_tensor(int(self.observation_size/2))
        self.obseveration_dx = self.observation_dx
        self.obseveration_dy = self.observation_dy
        self.zero_observation = self.float_zeros((self.nr_agents, self.nr_channels, self.observation_size, self.observation_size))
        self.one_observation = self.float_ones((self.nr_agents, self.observation_size, self.observation_size))
        self.current_position_map = -self.int_ones_like(self.obstacle_map)
        self.next_position_map = -self.int_ones_like(self.obstacle_map)
        self.vertex_collision_buffer = self.bool_ones(self.nr_agents)
        self.edge_collision_buffer = self.bool_ones(self.nr_agents)
        self.center_mask = self.float_zeros((self.nr_agents, self.observation_size, self.observation_size))
        half_size = int(self.observation_size/2)
        self.center_mask[:,half_size, half_size] = 1.0
        self.center_mask[:,half_size+1, half_size] = 1.0
        self.center_mask[:,half_size-1, half_size] = 1.0
        self.center_mask[:,half_size, half_size+1] = 1.0
        self.center_mask[:,half_size, half_size-1] = 1.0

    def joint_observation(self):
        obs = super(MAPFGridWorld, self).joint_observation()\
            .view(self.nr_agents, self.nr_channels, self.observation_size, self.observation_size)
        obs[:] = 0
        current_occupied_cells = self.populate_position_map(self.current_position_map, self.current_positions)
        goal_position_map, _ = self.get_pose_position_map(self.goal_positions)
        half_size = int(self.observation_size/2)
        current_anchor_positions = self.anchor_positions(self.current_positions)
        goal_anchor_positions = self.anchor_positions(self.goal_positions)
        x0 = current_anchor_positions[:,0]
        y0 = current_anchor_positions[:,1]
        x1 = goal_anchor_positions[:,0]
        y1 = goal_anchor_positions[:,1]
        dx = x1 - x0
        dy = y1 - y0
        abs_dx = torch.abs(dx)
        abs_dy = torch.abs(dy)
        manhattan_distance = abs_dx + abs_dy
        euclidean_distance = torch.sqrt(dx*dx + dy*dy)
        safe_euclidean_distance = torch.where(euclidean_distance > 0, euclidean_distance, self.float_ones_like(euclidean_distance))
        max_distance = torch.maximum(abs_dx, abs_dy)
        goal_in_sight = max_distance <= half_size

        # Scan position relative to the goal
        x_direction = torch.sign(dx).to(dtype=INT_TYPE)+half_size
        y_direction = torch.sign(dy).to(dtype=INT_TYPE)+half_size
        obs[self.agent_ids,MAPF_GOAL_DIRECTION_CHANNEL, x_direction, half_size] = abs_dx/safe_euclidean_distance
        obs[self.agent_ids,MAPF_GOAL_DIRECTION_CHANNEL, half_size, y_direction] = abs_dy/safe_euclidean_distance
        obs[self.agent_ids,MAPF_GOAL_DIRECTION_CHANNEL, half_size, half_size] = manhattan_distance.to(dtype=FLOAT_TYPE)
        obs[goal_in_sight,MAPF_OWN_GOAL_CHANNEL,dx[goal_in_sight]+half_size, dy[goal_in_sight]+half_size] = 1

        # Scan surrounding obstacles and boundaries
        dx = (x0.unsqueeze(1) + self.observation_dx).view(-1)
        dy = (y0.unsqueeze(1) + self.observation_dy).view(-1)
        in_bounds = self.xy_position_in_bounds(dx, dy).view(self.nr_agents, self.observation_size, self.observation_size)
        flat_in_bounds = in_bounds.view(-1)
        obs[self.agent_ids,MAPF_BLOCKED_CHANNEL,:,:] = torch.where(in_bounds, 0.0, 1.0)

        # Scan surrounding agent footprints and their manhattan distances to their goals
        x_clamped = dx.clamp(0, self.rows-1)
        y_clamped = dy.clamp(0, self.columns-1)
        observer_ids = self.agent_ids.unsqueeze(1)\
            .expand(-1, self.observation_size*self.observation_size)\
            .reshape(-1)

        zero_obs = self.zero_observation[:,MAPF_GOAL_DIRECTION_CHANNEL,:,:]
        neighbor_ids = self.current_position_map[x_clamped, y_clamped]
        is_agents_position = torch.logical_and(neighbor_ids >= 0, flat_in_bounds)
        is_other_agent_position = torch.logical_and(is_agents_position, neighbor_ids != observer_ids)
        if is_other_agent_position.any().item():
            neighbor_condition = is_other_agent_position.view(self.nr_agents, self.observation_size, self.observation_size)
            obs[self.agent_ids,MAPF_OTHER_AGENT_DISTANCE_CHANNEL,:,:] = torch.where(neighbor_condition, 1.0, 0.0)
            flattened_view = obs[self.agent_ids,MAPF_OTHER_AGENT_DISTANCE_CHANNEL].view(-1)
            flattened_view[neighbor_condition.view(-1)] = manhattan_distance[neighbor_ids[is_other_agent_position]].to(FLOAT_TYPE) + 1
            obs[self.agent_ids,MAPF_OTHER_AGENT_DISTANCE_CHANNEL,:,:] = flattened_view.view(self.nr_agents, self.observation_size, self.observation_size)
            obs[self.agent_ids,MAPF_GOAL_DIRECTION_CHANNEL,:,:] = torch.where(obs[self.agent_ids,MAPF_OTHER_AGENT_DISTANCE_CHANNEL,:,:] > 0, zero_obs, obs[self.agent_ids,MAPF_GOAL_DIRECTION_CHANNEL,:,:])
            obs[self.agent_ids,MAPF_BLOCKED_CHANNEL,:,:] = torch.where(obs[self.agent_ids,MAPF_OTHER_AGENT_DISTANCE_CHANNEL,:,:] > 0, self.one_observation, obs[self.agent_ids,MAPF_BLOCKED_CHANNEL,:,:])
            obs[self.agent_ids,self.other_footprint_channel,:,:] = torch.where(neighbor_condition, 1.0, 0.0)
            self.add_other_orientation_channels(obs, neighbor_ids, is_other_agent_position)

        self.add_self_pose_channels(obs, current_occupied_cells, half_size)

        # Scan surrounding goal footprints and their manhattan distances to their respective agents
        goal_ids = goal_position_map[x_clamped, y_clamped]
        is_goal_position = torch.logical_and(goal_ids >= 0, flat_in_bounds)
        is_other_goal_position = torch.logical_and(is_goal_position, goal_ids != observer_ids)
        if is_other_goal_position.any().item():
            goal_condition = is_other_goal_position.view(self.nr_agents, self.observation_size, self.observation_size)
            obs[self.agent_ids,MAPF_OTHER_GOAL_DISTANCE_CHANNEL,:,:] = torch.where(goal_condition, 1.0, 0.0)
            flattened_view = obs[self.agent_ids,MAPF_OTHER_GOAL_DISTANCE_CHANNEL].view(-1)
            distances = manhattan_distance[goal_ids[is_other_goal_position]].to(FLOAT_TYPE) + 1
            flattened_view[goal_condition.view(-1)] = distances
            obs[self.agent_ids,MAPF_OTHER_GOAL_DISTANCE_CHANNEL,:,:] = flattened_view.view(self.nr_agents, self.observation_size, self.observation_size)
            obs[self.agent_ids, MAPF_OTHER_GOAL_DISTANCE_CHANNEL, :, :] -= obs[self.agent_ids, MAPF_OWN_GOAL_CHANNEL, :, :]
            template = obs[self.agent_ids, MAPF_OTHER_GOAL_DISTANCE_CHANNEL, :, :]
            obs[self.agent_ids, MAPF_OTHER_GOAL_DISTANCE_CHANNEL, :, :] = torch.maximum(template, self.float_zeros_like(template))
        self.add_action_feasibility_channels(obs)
        return obs

    def get_pose_position_map(self, poses):
        position_map = -self.int_ones_like(self.obstacle_map)
        occupied_cells = self.occupied_cells_from_poses(poses)
        for agent_id, cells in enumerate(occupied_cells):
            position_map[cells[:,0], cells[:,1]] = agent_id
        return position_map, occupied_cells

    def add_self_pose_channels(self, obs, occupied_cells, half_size):
        current_orientations = self.pose_orientations(self.current_positions)
        for theta in range(self.nr_orientations):
            channel = self.self_orientation_channel + theta
            orientation_mask = (current_orientations == theta).to(FLOAT_TYPE).view(self.nr_agents, 1, 1)
            obs[:,channel,:,:] = orientation_mask.expand(-1, self.observation_size, self.observation_size)
        for agent_id, cells in enumerate(occupied_cells):
            local_offsets = cells - self.current_positions[agent_id,:ENV_2D].view(1, ENV_2D)
            local_x = local_offsets[:,0] + half_size
            local_y = local_offsets[:,1] + half_size
            visible = torch.logical_and(
                torch.logical_and(local_x >= 0, local_x < self.observation_size),
                torch.logical_and(local_y >= 0, local_y < self.observation_size)
            )
            if visible.any().item():
                obs[agent_id,self.self_footprint_channel,local_x[visible],local_y[visible]] = 1.0

    def add_other_orientation_channels(self, obs, neighbor_ids, is_other_agent_position):
        visible_orientations = -self.int_ones_like(neighbor_ids)
        visible_agent_ids = neighbor_ids[is_other_agent_position]
        visible_orientations[is_other_agent_position] = self.pose_orientations(self.current_positions)[visible_agent_ids]
        for theta in range(self.nr_orientations):
            channel = self.other_orientation_channel + theta
            theta_condition = (visible_orientations == theta).view(self.nr_agents, self.observation_size, self.observation_size)
            obs[:,channel,:,:] = torch.where(theta_condition, 1.0, 0.0)

    def add_action_feasibility_channels(self, obs):
        action_feasibility = self.get_action_feasibility()
        for action_index, _ in enumerate(ORIENTED_GRID_ACTIONS):
            channel = self.action_feasibility_channel + action_index
            feasibility = action_feasibility[:,action_index].view(self.nr_agents, 1, 1)
            obs[:,channel,:,:] = feasibility.expand(-1, self.observation_size, self.observation_size)

    def get_action_feasibility(self):
        action_feasibility = self.float_zeros((self.nr_agents, NR_ORIENTED_GRID_ACTIONS))
        for agent_id in range(self.nr_agents):
            for action_index, action in enumerate(ORIENTED_GRID_ACTIONS):
                action_feasibility[agent_id,action_index] = 1.0 if self.action_is_feasible(agent_id, action) else 0.0
        return action_feasibility

    def action_is_feasible(self, agent_id, action):
        if action == WAIT:
            return True
        candidate_pose = self.current_positions[agent_id].clone()
        heading = self.orientation_deltas[candidate_pose[ENV_2D]]
        if action == FORWARD:
            candidate_pose[:ENV_2D] += heading
        elif action == BACKWARD:
            candidate_pose[:ENV_2D] -= heading
        elif action == STRAFE_LEFT:
            candidate_pose[:ENV_2D] += self.left_heading_deltas(candidate_pose[ENV_2D].view(1))[0]
        elif action == STRAFE_RIGHT:
            candidate_pose[:ENV_2D] += self.right_heading_deltas(candidate_pose[ENV_2D].view(1))[0]
        elif action == ROTATE_LEFT:
            candidate_pose[ENV_2D] = torch.remainder(candidate_pose[ENV_2D] + 1, self.nr_orientations)
        elif action == ROTATE_RIGHT:
            candidate_pose[ENV_2D] = torch.remainder(candidate_pose[ENV_2D] - 1, self.nr_orientations)
        else:
            raise ValueError(f"Unknown oriented action: {action}")
        if not self.pose_is_valid(candidate_pose):
            return False
        candidate_cells = self.occupied_cells_from_pose(candidate_pose)
        occupant_ids = self.current_position_map[candidate_cells[:,0], candidate_cells[:,1]]
        overlaps_other_agent = torch.logical_and(occupant_ids >= 0, occupant_ids != agent_id).any()
        return not overlaps_other_agent.item()

    def get_delta_tensor(self, delta):
        assert delta > 0
        x = []
        y = []
        for _ in range(self.nr_agents):
            for dx in range(-delta, delta+1):
                for dy in range(-delta, delta+1):
                    x.append(dx)
                    y.append(dy)
        return self.as_int_tensor(x).view(self.nr_agents, -1),\
               self.as_int_tensor(y).view(self.nr_agents, -1)
