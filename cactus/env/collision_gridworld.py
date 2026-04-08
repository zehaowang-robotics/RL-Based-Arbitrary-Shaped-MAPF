from cactus.env.gridworld import GridWorld
from cactus.constants import *

"""
 Represents a navigation task in a 2D grid world environment with collision detection.
"""
class CollisionGridWorld(GridWorld):

    def __init__(self, params) -> None:
        super(CollisionGridWorld, self).__init__(params)
        self.agent_ids = self.as_int_tensor([i for i in range(self.nr_agents)])
        self.vertex_collision_buffer = self.bool_ones(self.nr_agents)
        self.edge_collision_buffer = self.bool_ones(self.nr_agents)

    def move_condition(self, new_positions):
        self.vertex_collision_buffer.fill_(False)
        self.edge_collision_buffer.fill_(False)
        condition, _ = super(CollisionGridWorld, self).move_condition(new_positions)
        condition = condition.all(1)
        candidate_positions = torch.where(condition.unsqueeze(1), new_positions, self.current_positions)
        source_cells = self.populate_position_map(self.current_position_map, self.current_positions)
        target_cells = self.populate_position_map(self.next_position_map, candidate_positions)

        cell_to_agents = {}
        for agent_id, cells in enumerate(target_cells):
            for cell in cells.tolist():
                key = (int(cell[0]), int(cell[1]))
                cell_to_agents.setdefault(key, []).append(agent_id)
        for agent_ids in cell_to_agents.values():
            if len(agent_ids) > 1:
                self.vertex_collision_buffer[agent_ids] = True

        moved = (candidate_positions != self.current_positions).any(1)
        source_sets = [set(map(tuple, cells.tolist())) for cells in source_cells]
        target_sets = [set(map(tuple, cells.tolist())) for cells in target_cells]
        for i in range(self.nr_agents):
            if not moved[i]:
                continue
            for j in range(i + 1, self.nr_agents):
                if not moved[j]:
                    continue
                exchanged_cells = target_sets[i].intersection(source_sets[j]) and target_sets[j].intersection(source_sets[i])
                if exchanged_cells:
                    self.edge_collision_buffer[i] = True
                    self.edge_collision_buffer[j] = True

        no_collisions = torch.logical_not(torch.logical_or(self.vertex_collision_buffer, self.edge_collision_buffer))
        condition = torch.logical_and(condition, no_collisions)
        return condition.unsqueeze(1).expand_as(new_positions), (self.vertex_collision_buffer, self.edge_collision_buffer)

    def reset(self):
        self.current_position_map[:] = -1.0
        self.next_position_map[:] = -1.0
        return super(CollisionGridWorld, self).reset()