from cactus.env.environment import Environment
from cactus.utils import assertContains, assertEquals, get_param_or_default
from cactus.constants import *
from cactus.rendering.gridworld_viewer import render
import torch
import random
import heapq
import math

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
        self.curriculum_radius_mode = get_param_or_default(params, CURRICULUM_RADIUS_MODE, CURRICULUM_RADIUS_ANCHOR_CHEBYSHEV)
        valid_radius_modes = {
            CURRICULUM_RADIUS_ANCHOR_CHEBYSHEV,
            CURRICULUM_RADIUS_SWEPT_AREA_CHEBYSHEV,
        }
        assert self.curriculum_radius_mode in valid_radius_modes, f"Unsupported curriculum radius mode: {self.curriculum_radius_mode}"
        self.relative_footprint_bounds = {}
        self.max_footprint_padding_x = 0
        self.max_footprint_padding_y = 0
        for theta in range(self.nr_orientations):
            bounds = self.compute_relative_footprint_bounds(theta)
            self.relative_footprint_bounds[theta] = bounds
            self.max_footprint_padding_x = max(self.max_footprint_padding_x, abs(bounds[0]), abs(bounds[1]))
            self.max_footprint_padding_y = max(self.max_footprint_padding_y, abs(bounds[2]), abs(bounds[3]))
        self.rotation_swept_footprints = self.get_rotation_swept_footprints()
        self.valid_pose_orientations = self.get_valid_pose_orientations()
        self.valid_anchor_positions = list(self.valid_pose_orientations.keys())
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
        self.init_goal_radius = None
        self.completion_reward = get_param_or_default(params, ENV_COMPLETION_REWARD, 1.0)
        self.use_primal_reward = get_param_or_default(params, ENV_USE_PRIMAL_REWARD, False)
        if self.use_primal_reward:
            self.collision_weight = 2
            self.time_penalty = -0.3
            self.completion_reward = 20.0
        self.set_init_goal_radius(get_param_or_default(params, ENV_INIT_GOAL_RADIUS, None))
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

    def rotated_cell_polygon(self, offset, angle):
        dx, dy = int(offset[0]), int(offset[1])
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        polygon = []
        for x, y in (
            (dx - 0.5, dy - 0.5),
            (dx + 0.5, dy - 0.5),
            (dx + 0.5, dy + 0.5),
            (dx - 0.5, dy + 0.5),
        ):
            polygon.append((
                x * cos_angle - y * sin_angle,
                x * sin_angle + y * cos_angle,
            ))
        return polygon

    def polygon_area(self, polygon):
        if len(polygon) < 3:
            return 0.0
        area = 0.0
        for index, point in enumerate(polygon):
            next_point = polygon[(index + 1) % len(polygon)]
            area += point[0] * next_point[1] - next_point[0] * point[1]
        return abs(area) * 0.5

    def clip_polygon_half_plane(self, polygon, inside, intersection):
        if not polygon:
            return []
        clipped = []
        previous = polygon[-1]
        previous_inside = inside(previous)
        for current in polygon:
            current_inside = inside(current)
            if current_inside:
                if not previous_inside:
                    clipped.append(intersection(previous, current))
                clipped.append(current)
            elif previous_inside:
                clipped.append(intersection(previous, current))
            previous = current
            previous_inside = current_inside
        return clipped

    def clip_polygon_to_cell(self, polygon, cell):
        x, y = int(cell[0]), int(cell[1])
        left = x - 0.5
        right = x + 0.5
        bottom = y - 0.5
        top = y + 0.5
        eps = 1e-12
        clipped = polygon
        clipped = self.clip_polygon_half_plane(
            clipped,
            lambda point: point[0] >= left - eps,
            lambda start, end: (
                left,
                start[1] + (end[1] - start[1]) * (left - start[0]) / (end[0] - start[0]),
            )
        )
        clipped = self.clip_polygon_half_plane(
            clipped,
            lambda point: point[0] <= right + eps,
            lambda start, end: (
                right,
                start[1] + (end[1] - start[1]) * (right - start[0]) / (end[0] - start[0]),
            )
        )
        clipped = self.clip_polygon_half_plane(
            clipped,
            lambda point: point[1] >= bottom - eps,
            lambda start, end: (
                start[0] + (end[0] - start[0]) * (bottom - start[1]) / (end[1] - start[1]),
                bottom,
            )
        )
        clipped = self.clip_polygon_half_plane(
            clipped,
            lambda point: point[1] <= top + eps,
            lambda start, end: (
                start[0] + (end[0] - start[0]) * (top - start[1]) / (end[1] - start[1]),
                top,
            )
        )
        return clipped

    def cells_overlapping_polygon(self, polygon):
        min_x = min(point[0] for point in polygon)
        max_x = max(point[0] for point in polygon)
        min_y = min(point[1] for point in polygon)
        max_y = max(point[1] for point in polygon)
        eps = 1e-9
        min_cell_x = math.floor(min_x + 0.5 + eps)
        max_cell_x = math.ceil(max_x - 0.5 - eps)
        min_cell_y = math.floor(min_y + 0.5 + eps)
        max_cell_y = math.ceil(max_y - 0.5 - eps)
        cells = []
        for x in range(min_cell_x, max_cell_x + 1):
            for y in range(min_cell_y, max_cell_y + 1):
                clipped = self.clip_polygon_to_cell(polygon, (x, y))
                if self.polygon_area(clipped) > eps:
                    cells.append((x, y))
        return cells

    def compute_rotation_swept_footprint(self, start_theta, rotation_direction):
        swept_offsets = set()
        steps = 90
        start_angle = int(start_theta) * math.pi / 2.0
        for step in range(steps + 1):
            angle = start_angle + int(rotation_direction) * (math.pi / 2.0) * step / steps
            for offset in self.agent_footprint:
                polygon = self.rotated_cell_polygon(offset, angle)
                swept_offsets.update(self.cells_overlapping_polygon(polygon))
        end_theta = (int(start_theta) + int(rotation_direction)) % self.nr_orientations
        swept_offsets.update(tuple(cell) for cell in self.transformed_footprint(start_theta).tolist())
        swept_offsets.update(tuple(cell) for cell in self.transformed_footprint(end_theta).tolist())
        return self.as_int_tensor(sorted(swept_offsets)).view(-1, ENV_2D)

    def get_rotation_swept_footprints(self):
        swept_footprints = {}
        for theta in range(self.nr_orientations):
            left_theta = (theta + 1) % self.nr_orientations
            right_theta = (theta - 1) % self.nr_orientations
            swept_footprints[(theta, left_theta)] = self.compute_rotation_swept_footprint(theta, 1)
            swept_footprints[(theta, right_theta)] = self.compute_rotation_swept_footprint(theta, -1)
        return swept_footprints

    def occupied_cells_from_pose(self, pose, footprint=None):
        pose_tensor = self.as_pose(pose)
        footprint_offsets = self.transformed_footprint(pose_tensor[ENV_2D].item(), footprint)
        return footprint_offsets + pose_tensor[:ENV_2D].view(1, ENV_2D)

    def occupied_cells_from_poses(self, poses, footprint=None):
        pose_batch = self.as_pose_batch(poses)
        occupied_cells = [self.occupied_cells_from_pose(pose, footprint) for pose in pose_batch]
        return self.stack(occupied_cells)

    def pose_cells_as_tuples(self, pose, footprint=None):
        return tuple((int(cell[0]), int(cell[1])) for cell in self.occupied_cells_from_pose(pose, footprint).tolist())

    def is_rotation_transition(self, source_pose, target_pose):
        source_pose = self.as_pose(source_pose)
        target_pose = self.as_pose(target_pose)
        if not torch.equal(source_pose[:ENV_2D], target_pose[:ENV_2D]):
            return False
        orientation_delta = int(torch.remainder(target_pose[ENV_2D] - source_pose[ENV_2D], self.nr_orientations).item())
        return orientation_delta == 1 or orientation_delta == self.nr_orientations - 1

    def transition_cells_from_pose(self, source_pose, target_pose):
        source_pose = self.as_pose(source_pose)
        target_pose = self.as_pose(target_pose)
        if self.is_rotation_transition(source_pose, target_pose):
            key = (int(source_pose[ENV_2D].item()), int(target_pose[ENV_2D].item()))
            footprint_offsets = self.rotation_swept_footprints[key]
            return footprint_offsets + source_pose[:ENV_2D].view(1, ENV_2D)
        return self.occupied_cells_from_pose(target_pose)

    def transition_cells_from_poses(self, source_poses, target_poses):
        source_batch = self.as_pose_batch(source_poses)
        target_batch = self.as_pose_batch(target_poses)
        return [
            self.transition_cells_from_pose(source_pose, target_pose)
            for source_pose, target_pose in zip(source_batch, target_batch)
        ]

    def cells_are_valid(self, cells):
        x = cells[:,0]
        y = cells[:,1]
        return bool(self.xy_position_in_bounds(x, y).all().item())

    def compute_relative_footprint_bounds(self, theta, footprint=None):
        footprint_offsets = self.transformed_footprint(theta, footprint)
        x = footprint_offsets[:,0].tolist()
        y = footprint_offsets[:,1].tolist()
        return min(x), max(x), min(y), max(y)

    def pose_bounds(self, pose, footprint=None):
        pose_tensor = self.as_pose(pose)
        theta = int(pose_tensor[ENV_2D].item())
        if footprint is None:
            min_dx, max_dx, min_dy, max_dy = self.relative_footprint_bounds[theta]
        else:
            min_dx, max_dx, min_dy, max_dy = self.compute_relative_footprint_bounds(theta, footprint)
        x = int(pose_tensor[0].item())
        y = int(pose_tensor[1].item())
        return x + min_dx, x + max_dx, y + min_dy, y + max_dy

    def pose_is_valid(self, pose, footprint=None):
        occupied_cells = self.occupied_cells_from_pose(pose, footprint)
        return self.cells_are_valid(occupied_cells)

    def get_valid_pose_orientations(self):
        valid_pose_orientations = {}
        for anchor in self.occupiable_locations:
            x, y = anchor
            valid_orientations = []
            for theta in range(self.nr_orientations):
                pose = self.as_pose([x, y, theta])
                if self.pose_is_valid(pose):
                    valid_orientations.append(theta)
            if valid_orientations:
                valid_pose_orientations[anchor] = tuple(valid_orientations)
        return valid_pose_orientations

    def cells_are_available(self, cells, occupied_cells):
        return all(cell not in occupied_cells for cell in cells)

    def enumerate_valid_anchors_in_box(self, min_x, max_x, min_y, max_y):
        anchors = []
        min_x = max(0, int(min_x))
        max_x = min(self.rows - 1, int(max_x))
        min_y = max(0, int(min_y))
        max_y = min(self.columns - 1, int(max_y))
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                anchor = (x, y)
                if anchor in self.valid_pose_orientations:
                    anchors.append(anchor)
        return anchors

    def goal_radius_is_unbounded(self, radius):
        return radius is None or int(radius) < 0

    def anchor_goal_radius(self, start_pose, goal_pose):
        start_pose = self.as_pose(start_pose)
        goal_pose = self.as_pose(goal_pose)
        dx = abs(int(goal_pose[0].item()) - int(start_pose[0].item()))
        dy = abs(int(goal_pose[1].item()) - int(start_pose[1].item()))
        return max(dx, dy)

    def swept_area_goal_radius(self, start_pose, goal_pose):
        start_min_x, start_max_x, start_min_y, start_max_y = self.pose_bounds(start_pose)
        goal_min_x, goal_max_x, goal_min_y, goal_max_y = self.pose_bounds(goal_pose)
        return max(
            max(0, start_min_x - goal_min_x),
            max(0, goal_max_x - start_max_x),
            max(0, start_min_y - goal_min_y),
            max(0, goal_max_y - start_max_y),
        )

    def goal_radius(self, start_pose, goal_pose):
        if self.curriculum_radius_mode == CURRICULUM_RADIUS_ANCHOR_CHEBYSHEV:
            return self.anchor_goal_radius(start_pose, goal_pose)
        if self.curriculum_radius_mode == CURRICULUM_RADIUS_SWEPT_AREA_CHEBYSHEV:
            return self.swept_area_goal_radius(start_pose, goal_pose)
        raise ValueError(f"Unsupported curriculum radius mode: {self.curriculum_radius_mode}")

    def goal_pose_within_radius(self, start_pose, goal_pose, radius):
        if self.goal_radius_is_unbounded(radius):
            return True
        return self.goal_radius(start_pose, goal_pose) <= int(radius)

    def goal_candidate_anchor_bounds(self, start_pose, radius):
        radius = int(radius)
        if self.curriculum_radius_mode == CURRICULUM_RADIUS_ANCHOR_CHEBYSHEV:
            pose_tensor = self.as_pose(start_pose)
            x = int(pose_tensor[0].item())
            y = int(pose_tensor[1].item())
            return x - radius, x + radius, y - radius, y + radius
        if self.curriculum_radius_mode == CURRICULUM_RADIUS_SWEPT_AREA_CHEBYSHEV:
            min_x, max_x, min_y, max_y = self.pose_bounds(start_pose)
            return (
                min_x - radius - self.max_footprint_padding_x,
                max_x + radius + self.max_footprint_padding_x,
                min_y - radius - self.max_footprint_padding_y,
                max_y + radius + self.max_footprint_padding_y,
            )
        raise ValueError(f"Unsupported curriculum radius mode: {self.curriculum_radius_mode}")

    def get_goal_candidate_anchors(self, start_pose, radius):
        if self.goal_radius_is_unbounded(radius):
            anchors = list(self.valid_anchor_positions)
            random.shuffle(anchors)
            return anchors
        anchors = self.enumerate_valid_anchors_in_box(*self.goal_candidate_anchor_bounds(start_pose, radius))
        random.shuffle(anchors)
        return anchors

    def sample_start_pose_batch(self):
        anchors = list(self.valid_anchor_positions)
        random.shuffle(anchors)
        sampled_poses = []
        occupied_cells = set()
        for anchor in anchors:
            valid_orientations = list(self.valid_pose_orientations[anchor])
            random.shuffle(valid_orientations)
            for theta in valid_orientations:
                pose = self.as_pose([anchor[0], anchor[1], theta])
                cells = self.pose_cells_as_tuples(pose)
                if self.cells_are_available(cells, occupied_cells):
                    sampled_poses.append(pose)
                    occupied_cells.update(cells)
                    break
            if len(sampled_poses) == self.nr_agents:
                return self.stack(sampled_poses)
        return None

    def sample_goal_pose_batch(self, start_poses):
        goal_poses = [None for _ in range(self.nr_agents)]
        occupied_goal_cells = set()
        occupied_goal_anchors = set()
        agent_ids = list(range(self.nr_agents))
        random.shuffle(agent_ids)
        for agent_id in agent_ids:
            start_pose = self.as_pose(start_poses[agent_id])
            start_anchor = tuple(int(v) for v in start_pose[:ENV_2D].tolist())
            sampled_goal_pose = None
            for anchor in self.get_goal_candidate_anchors(start_pose, self.init_goal_radius):
                if anchor == start_anchor or anchor in occupied_goal_anchors:
                    continue
                valid_orientations = list(self.valid_pose_orientations[anchor])
                random.shuffle(valid_orientations)
                for theta in valid_orientations:
                    goal_pose = self.as_pose([anchor[0], anchor[1], theta])
                    if not self.goal_pose_within_radius(start_pose, goal_pose, self.init_goal_radius):
                        continue
                    goal_cells = self.pose_cells_as_tuples(goal_pose)
                    if not self.cells_are_available(goal_cells, occupied_goal_cells):
                        continue
                    sampled_goal_pose = goal_pose
                    occupied_goal_cells.update(goal_cells)
                    occupied_goal_anchors.add(anchor)
                    break
                if sampled_goal_pose is not None:
                    break
            if sampled_goal_pose is None:
                return None
            goal_poses[agent_id] = sampled_goal_pose
        return self.stack(goal_poses)

    def validate_pose_batch(self, positions, require_unique_anchors=False, label="poses"):
        poses = self.as_pose_batch(positions)
        occupied_cells = set()
        occupied_anchors = set()
        for pose in poses:
            if not self.pose_is_valid(pose):
                raise ValueError(f"Invalid {label}: footprint leaves the map or overlaps obstacles")
            cells = self.pose_cells_as_tuples(pose)
            if not self.cells_are_available(cells, occupied_cells):
                raise ValueError(f"Invalid {label}: footprint overlap detected")
            occupied_cells.update(cells)
            if require_unique_anchors:
                anchor = tuple(int(v) for v in pose[:ENV_2D].tolist())
                if anchor in occupied_anchors:
                    raise ValueError(f"Invalid {label}: duplicate anchor detected")
                occupied_anchors.add(anchor)
        return poses

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

    def left_heading_deltas(self, orientations):
        headings = self.heading_deltas(orientations)
        left_headings = self.int_zeros_like(headings)
        left_headings[:,0] = -headings[:,1]
        left_headings[:,1] = headings[:,0]
        return left_headings

    def right_heading_deltas(self, orientations):
        return -self.left_heading_deltas(orientations)

    def transition_poses(self, joint_action):
        next_positions = self.current_positions.clone()
        orientations = self.pose_orientations(self.current_positions)
        headings = self.heading_deltas(orientations)
        left_headings = self.left_heading_deltas(orientations)
        right_headings = self.right_heading_deltas(orientations)
        forward_mask = joint_action == FORWARD
        backward_mask = joint_action == BACKWARD
        strafe_left_mask = joint_action == STRAFE_LEFT
        strafe_right_mask = joint_action == STRAFE_RIGHT
        rotate_left_mask = joint_action == ROTATE_LEFT
        rotate_right_mask = joint_action == ROTATE_RIGHT
        next_positions[forward_mask, :ENV_2D] += headings[forward_mask]
        next_positions[backward_mask, :ENV_2D] -= headings[backward_mask]
        next_positions[strafe_left_mask, :ENV_2D] += left_headings[strafe_left_mask]
        next_positions[strafe_right_mask, :ENV_2D] += right_headings[strafe_right_mask]
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

    def transition_in_bounds(self, new_positions):
        transition_cells = self.transition_cells_from_poses(self.current_positions, new_positions)
        return self.as_bool_tensor([self.cells_are_valid(cells) for cells in transition_cells])

    def move_condition(self, new_positions):
        in_bounds = self.transition_in_bounds(new_positions)
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
        if radius is None:
            self.init_goal_radius = None
            return
        radius = int(radius)
        self.init_goal_radius = None if radius < 0 else radius

    def increment_init_goal_radius(self):
        if self.init_goal_radius is None:
            self.init_goal_radius = 0
            return
        self.init_goal_radius += 1

    def decrement_init_goal_radius(self):
        if self.init_goal_radius is None:
            return
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
        self.current_positions[:] = self.validate_pose_batch(positions, label="start poses")
        self.populate_position_map(self.current_position_map, self.current_positions)

    def set_goal_positions(self, positions):
        poses = self.validate_pose_batch(positions, require_unique_anchors=True, label="goal poses")
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
        for _ in range(max(32, self.nr_agents * 8)):
            sampled_start_poses = self.sample_start_pose_batch()
            if sampled_start_poses is None:
                continue
            sampled_goal_poses = self.sample_goal_pose_batch(sampled_start_poses)
            if sampled_goal_poses is None:
                continue
            self.set_start_positions(sampled_start_poses)
            self.set_goal_positions(sampled_goal_poses)
            assertEquals(self.nr_agents, len(self.occupied_goal_positions[self.occupied_goal_positions >= 0].view(-1)))
            return self.joint_observation()
        raise RuntimeError("Failed to sample a non-overlapping start/goal pose configuration")

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
