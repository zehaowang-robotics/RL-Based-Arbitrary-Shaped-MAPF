# Oriented Multi-Cell Agent Modeling Spec

This document freezes the modeling choices for the first implementation of multi-cell agents with rotation.

## Scope

This spec applies to the oriented multi-cell environment refactor. The current training default uses an L-shaped three-cell footprint unless overridden.

## Frozen Decisions

### 1. Agent state representation

Each agent state is represented as a pose:

`pose_i = (x_i, y_i, theta_i)`

- `(x_i, y_i)` is the anchor cell in map coordinates.
- `theta_i` is a discrete orientation index.
- `ENV_POSE_DIM = 3`.

### 2. Orientation set

Orientation is discrete and limited to four right-angle rotations:

- `THETA_0`
- `THETA_90`
- `THETA_180`
- `THETA_270`

Interpretation:

- `theta = 0` means the canonical footprint orientation.
- Rotations are applied counter-clockwise in 90 degree increments.
- `ENV_NR_ORIENTATIONS` defaults to `4`.

This keeps the environment compatible with grid-based occupancy and avoids continuous-rotation geometry.

### 3. Footprint format

Each agent footprint is defined in local coordinates relative to the anchor at `theta = 0`.

Format:

`ENV_AGENT_FOOTPRINT = ((dx_1, dy_1), ..., (dx_k, dy_k))`

Rules:

- The anchor cell is part of the footprint.
- Local coordinates are integer grid offsets.
- The current default is an L-shaped three-cell agent with the anchor/state point at `(0, 0)`:
  `DEFAULT_AGENT_FOOTPRINT = ((0, 0), (0, 1), (1, 0))`
- Rotated occupancies are obtained by rotating each local offset around the anchor.

Example 2x1 body:

`((0, 0), (1, 0))`

### 4. Action set

The oriented action space is:

- `WAIT`
- `FORWARD`
- `BACKWARD`
- `ROTATE_LEFT`
- `ROTATE_RIGHT`
- `STRAFE_LEFT`
- `STRAFE_RIGHT`

The strafe actions translate the anchor one cell left or right relative to the current heading without changing orientation.

Rationale:

- It supports forward/backward movement while still allowing lateral repositioning.
- It keeps rotation as a separate in-place action.
- It exposes a fixed action count of 7 to the policy and action mask.

### 5. Motion semantics

- `FORWARD` and `BACKWARD` move the anchor by one grid cell along the agent heading.
- `STRAFE_LEFT` and `STRAFE_RIGHT` move the anchor by one grid cell perpendicular to the heading without changing orientation.
- `ROTATE_LEFT` and `ROTATE_RIGHT` rotate the footprint in place around the anchor.
- A rotation transition is valid only if every grid cell swept by the footprint during the quarter-turn is in bounds, obstacle free, and collision free.
- A non-rotation transition remains valid only if every occupied cell after the action is in bounds, obstacle free, and collision free.

### 6. Occupancy model

An agent occupies all map cells covered by its transformed footprint.

This means:

- obstacle checks operate on the full occupied cell set
- agent-agent collision checks operate on the full occupied cell set
- swap and edge-conflict logic must be generalized from point occupancy to footprint occupancy

### 7. Goal definition

Phase 1 goal completion is anchor-position based only.

- `ENV_GOAL_ORIENTATION_REQUIRED = False`
- Goal success means the anchor reaches the goal anchor.
- Final orientation is not required in phase 1.

Rationale:

- It minimizes reward and evaluation changes.
- It keeps the first environment refactor focused on geometry and collisions.
- Orientation-sensitive goals can be added later behind a config flag.

### 8. Curriculum radius definition

Curriculum radius remains anchor based in phase 1.

- `CURRICULUM_RADIUS_MODE = "anchor_chebyshev"`
- `ENV_INIT_GOAL_RADIUS` is interpreted on the anchor position only
- the radius metric is the same square-neighborhood style currently used by reset sampling

Rationale:

- It is the least disruptive extension of the current CACTUS setup.
- It avoids introducing swept-volume or footprint-aware curriculum sampling before the new environment is stable.

### 9. Sampling defaults

When the environment is upgraded:

- start poses should sample both anchor and orientation
- goal poses should sample anchor only in phase 1
- orientation can still be stored for future compatibility, but it is not part of success unless explicitly enabled

## Config Surface Introduced In This Stage

The following constants are now reserved for the implementation work:

- `ENV_POSE_DIM`
- `ENV_START_POSES`
- `ENV_GOAL_POSES`
- `ENV_START_ORIENTATIONS`
- `ENV_GOAL_ORIENTATIONS`
- `ENV_NR_ORIENTATIONS`
- `ENV_AGENT_FOOTPRINT`
- `ENV_GOAL_ORIENTATION_REQUIRED`
- `CURRICULUM_RADIUS_MODE`
- `CURRICULUM_RADIUS_ANCHOR_CHEBYSHEV`

## Explicit Non-Goals For Phase 1

The first implementation does not include:

- continuous headings
- arbitrary-angle rotation
- orientation-sensitive success by default
- heterogeneous action spaces across agents
- footprint-aware swept-area curriculum metrics

## Implementation Order

The next stages should follow this dependency order:

1. pose and footprint data model
2. footprint-aware transition and collision logic
3. reset and curriculum sampling
4. observation redesign
5. action masking and training entrypoints
