import pygame

BLACK = (0, 0, 0)
DARK_GRAY = (125, 125, 125)
GRAY = (175, 175, 175)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
ORANGE = (255, 150, 0)
BLUE = (0, 0, 255)
LIGHT_BLUE = (51, 226, 253)
GREEN = (0, 255, 0)
MAGENTA = (255, 0, 255)
MAROON = (128, 0, 0)
CYAN = (0, 255, 255)
TEAL = (0, 128, 128)
PURPLE = (128, 0, 128)

AGENT_COLORS = [RED, BLUE, ORANGE, MAGENTA, PURPLE, TEAL, MAROON, GREEN, DARK_GRAY, CYAN]

class GridworldViewer:
    def __init__(self, width, height, cell_size=10, fps=30):
        pygame.init()
        self.cell_size = cell_size
        self.width = cell_size*width
        self.height = cell_size*height
        self.clock = pygame.time.Clock()
        self.fps = fps
        pygame.display.set_caption("MAPF Environment")
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.event.set_blocked(pygame.MOUSEMOTION)  # we do not need mouse movement events

    def agent_color(self, agent_id):
        nr_colors = len(AGENT_COLORS)
        return AGENT_COLORS[agent_id%nr_colors]

    def draw_state(self, env):
        self.screen.fill(BLACK)
        self.draw_grid(env)
        self.draw_goal_footprints(env)
        self.draw_agent_footprints(env)
        pygame.display.flip()
        self.clock.tick(self.fps)
        return self.check_for_interrupt()

    def draw_grid(self, env):
        for x in range(env.rows):
            for y in range(env.columns):
                if not env.obstacle_map[x][y]:
                    self.draw_pixel(x, y, WHITE)

    def draw_goal_footprints(self, env):
        if not hasattr(env, "goal_positions"):
            return
        goal_cells = env.occupied_cells_from_poses(env.goal_positions)
        goal_anchors = env.anchor_positions(env.goal_positions)
        for agent_id, cells in enumerate(goal_cells):
            color = self.agent_color(agent_id)
            for cell in cells.tolist():
                self.draw_cell_outline(int(cell[0]), int(cell[1]), color, width=2)
            anchor_x = int(goal_anchors[agent_id,0].item())
            anchor_y = int(goal_anchors[agent_id,1].item())
            self.draw_pixel(anchor_x, anchor_y, color, margin=3)

    def draw_agent_footprints(self, env):
        if not hasattr(env, "current_positions"):
            return
        occupied_cells = env.occupied_cells_from_poses(env.current_positions)
        anchors = env.anchor_positions(env.current_positions)
        headings = env.heading_deltas(env.pose_orientations(env.current_positions))
        for agent_id, cells in enumerate(occupied_cells):
            color = self.agent_color(agent_id)
            for cell in cells.tolist():
                self.draw_footprint_cell(int(cell[0]), int(cell[1]), color)
            anchor_x = int(anchors[agent_id,0].item())
            anchor_y = int(anchors[agent_id,1].item())
            heading_x = int(headings[agent_id,0].item())
            heading_y = int(headings[agent_id,1].item())
            self.draw_anchor_marker(anchor_x, anchor_y)
            self.draw_heading_marker(anchor_x, anchor_y, heading_x, heading_y)

    def cell_rect(self, x, y, margin=1):
        return pygame.Rect(
            x * self.cell_size + margin,
            y * self.cell_size + margin,
            self.cell_size - 2 * margin,
            self.cell_size - 2 * margin)

    def draw_pixel(self, x, y, color, margin=1):
        pygame.draw.rect(self.screen, color, self.cell_rect(x, y, margin), 0)

    def draw_cell_outline(self, x, y, color, width=1):
        pygame.draw.rect(self.screen, color, self.cell_rect(x, y, 1), width)

    def draw_footprint_cell(self, x, y, color):
        pygame.draw.rect(self.screen, BLACK, self.cell_rect(x, y, 0), 0)
        pygame.draw.rect(self.screen, color, self.cell_rect(x, y, 2), 0)

    def draw_anchor_marker(self, x, y):
        radius = max(2, int(self.cell_size/4))
        center = self.cell_center(x, y)
        pygame.draw.circle(self.screen, BLACK, center, radius+1)
        pygame.draw.circle(self.screen, WHITE, center, radius)

    def draw_heading_marker(self, x, y, dx, dy):
        start = self.cell_center(x, y)
        scale = max(2, int(self.cell_size/2) - 1)
        end = (start[0] + dx*scale, start[1] + dy*scale)
        pygame.draw.line(self.screen, BLACK, start, end, 2)
        pygame.draw.circle(self.screen, BLACK, end, max(2, int(self.cell_size/8)))

    def cell_center(self, x, y):
        radius = int(self.cell_size/2)
        return (x * self.cell_size + radius, y * self.cell_size + radius)

    def check_for_interrupt(self):
        key_state = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT or key_state[pygame.K_ESCAPE]:
                return True
        return False

    def close(self):
        pygame.quit()

def render(env, viewer):
    if viewer is None:
        viewer = GridworldViewer(env.columns, env.rows)
    viewer.draw_state(env)
    return viewer
