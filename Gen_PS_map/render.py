"""Matplotlib rendering of SocialNavEnv: ``draw(env, ax)``."""
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np

# Body fill = felt valence (red negative, grey neutral, green positive), opacity = intensity.
VALENCE_CMAP = mcolors.LinearSegmentedColormap.from_list("valence", ["#e8554e", "#b8b7b0", "#3fae7a"])
VALENCE_NORM = mcolors.Normalize(vmin=-1.0, vmax=1.0)

# Outline ring = response state (what the human is doing about the robot).
RESPONSE_COLORS = {
    'walk': 'white',
    'attend': '#2a78d6',
    'stop': '#eb6834',
    'sidestep': '#1baf7a',
    'startle': '#eda100',
}

PERSONALITY_MARKERS = {'regular': 'o', 'grumpy': 's', 'sad': '^', 'chill': 'D'}


def get_personality(stability, plasticity):
    if stability >= 0:
        return 'regular' if plasticity >= 0 else 'chill'
    return 'grumpy' if plasticity >= 0 else 'sad'


def get_asymmetric_proxemic_polygon(pos_x, pos_y, orientation, sigma_front, sigma_back,
                                    sigma_left, sigma_right, num_points=100):
    orientation = orientation or 0.0
    gamma = np.linspace(0, 2 * np.pi, num_points)
    sigma_long = np.where(np.cos(gamma) >= 0, sigma_front, sigma_back)
    sigma_lat = np.where(np.sin(gamma) >= 0, sigma_left, sigma_right)
    r = 1.0 / np.sqrt(np.cos(gamma) ** 2 / sigma_long ** 2 + np.sin(gamma) ** 2 / sigma_lat ** 2)
    lx, ly = r * np.cos(gamma), r * np.sin(gamma)
    c, s = np.cos(orientation), np.sin(orientation)
    return np.column_stack((pos_x + lx * c - ly * s, pos_y + lx * s + ly * c))


def _guarded_ps(agent):
    """Front/back/left/right as the agent's perception uses them: once aware
    of the robot, the rear is guarded as much as the sides."""
    back = max(agent.ps_back, agent.ps_left, agent.ps_right) if agent.aware else agent.ps_back
    return agent.ps_front, back, agent.ps_left, agent.ps_right


def _draw_fov(ax, agent, radius=1.2):
    """Faint wedge showing the field of view (only while not aware of the robot)."""
    if agent.orientation is None or agent.aware:
        return
    half, heading = np.degrees(agent.FOV_HALF_ANGLE), np.degrees(agent.orientation)
    ax.add_patch(mpatches.Wedge(tuple(agent.pos), radius, heading - half, heading + half,
                                facecolor='white', edgecolor='none', alpha=0.05, zorder=1))


def draw_force_field(env, ax, grid_n=15, extent=2.5):
    """Planner escape + goal forces sampled around the robot."""
    offsets = np.linspace(-extent, extent, grid_n)
    pts = np.array([env.robot.pos + (dx, dy) for dx in offsets for dy in offsets], dtype=np.float32)
    forces = np.array([env.planner.escape_forces(p) for p in pts])
    ax.quiver(pts[:, 0], pts[:, 1], forces[:, 0], forces[:, 1], color='cyan', alpha=0.7, scale=20, zorder=5)


def _draw_human(ax, agent, predicted_ps):
    x, y = agent.pos
    orientation = agent.orientation or 0.0
    ps = _guarded_ps(agent)
    valence, intensity = agent.emotion
    response = agent.response

    _draw_fov(ax, agent)
    # Personal space (solid) and the zone where the robot gets noticed / attended (dotted)
    ax.add_patch(mpatches.Polygon(get_asymmetric_proxemic_polygon(x, y, orientation, *ps), closed=True,
                                  facecolor='red', edgecolor='red', linestyle='--', alpha=0.15, zorder=2))
    ax.add_patch(mpatches.Polygon(
        get_asymmetric_proxemic_polygon(x, y, orientation, *(agent.ATTEND_ENTER * s for s in ps)),
        closed=True, fill=False, edgecolor='red', linestyle=':', linewidth=0.8, alpha=0.4, zorder=2))

    # The policy's predicted personal space [front, left, right, back]
    if predicted_ps is not None and np.any(predicted_ps):
        f, l, r, b = predicted_ps
        """"
        f, s, b = predicted_ps
        l, r = s, s  # symmetric left/right
        """
        ax.add_patch(mpatches.Polygon(get_asymmetric_proxemic_polygon(x, y, orientation, f, b, l, r), closed=True,
                                      facecolor='blue', edgecolor='blue', linestyle='--', alpha=0.15, zorder=2))

    # Body: fill = valence, opacity = intensity, ring = response state
    ax.add_patch(mpatches.Circle((x, y), agent.radius, facecolor=VALENCE_CMAP(VALENCE_NORM(valence)),
                                 edgecolor='none', alpha=0.45 + 0.55 * intensity, zorder=3))
    ax.add_patch(mpatches.Circle((x, y), agent.radius, fill=False, edgecolor=RESPONSE_COLORS.get(response, 'white'),
                                 linewidth=2.5 if response != 'walk' else 0.6, zorder=3))
    ax.quiver(x, y, 0.5 * np.cos(orientation), 0.5 * np.sin(orientation), angles='xy', scale_units='xy',
              scale=1, color='black', width=0.007, headwidth=4, headlength=5, zorder=4)
    if response == 'sidestep':
        dx, dy = 0.5 * agent.sidestep_dir
        ax.annotate('', xy=(x + dx, y + dy), xytext=(x, y), zorder=4,
                    arrowprops=dict(arrowstyle='->', color=RESPONSE_COLORS['sidestep'], lw=1.5))

    marker = PERSONALITY_MARKERS[get_personality(agent.stability, agent.plasticity)]
    ax.plot(x, y, marker=marker, color='none', markersize=6, markeredgecolor='black', markeredgewidth=0.6, zorder=5)
    ax.text(x, y, str(agent.group_id % 10), ha='center', va='center', fontsize=5, color='black', zorder=6)

    event = agent.behaviour_type if agent.behaviour_type != 'regular' else ''
    reaction = f" ({agent.startle_reaction})" if response == 'startle' and agent.startle_reaction else ''
    label = f"{response}{reaction}" + (f"\n{event.replace('_', ' ')}" if event else '') \
        + f"\nv {valence:+.2f}  i {intensity:.2f}"
    ax.text(x + 0.32, y + 0.32, label, fontsize=6, color='white', va='bottom', zorder=6,
            bbox=dict(facecolor='#1a1a2e', edgecolor='none', alpha=0.6, pad=1))


def draw(env, ax):
    ax.clear()
    ax.set_xlim(0.0, env.width)
    ax.set_ylim(0.0, env.height)
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a2e')
    ax.grid(True, color='white', linewidth=0.2, alpha=0.3)
    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))

    planner, robot = env.planner, env.robot
    if planner.costmap is not None:
        half = planner.MAP_SIZE / 2
        ax.imshow(planner.costmap, origin='lower', cmap='inferno', alpha=0.5, zorder=1,
                  extent=[robot.pos[0] - half, robot.pos[0] + half, robot.pos[1] - half, robot.pos[1] + half])

    observed_ids = list(env._obs['agents'][:, 7])
    #observed_ids = list(env._obs['agents'][:, 5])
    for agent in env.humans:
        predicted = None
        if env.action is not None and agent.human_id in observed_ids:
            predicted = env.action[observed_ids.index(agent.human_id)]
        _draw_human(ax, agent, predicted)

    for o in env.obstacles:
        ax.add_patch(mpatches.Circle(o.pos, o.radius, color='gray', alpha=0.6))

    ax.add_patch(mpatches.Circle(tuple(robot.pos), robot.radius, facecolor='orange', edgecolor='white',
                                 linewidth=0.8, alpha=0.85, zorder=3))
    ax.plot(*robot.pos, marker='*', color='orange', markersize=18, markeredgecolor='white',
            markeredgewidth=0.8, zorder=4)
    draw_force_field(env, ax)
    ax.plot(*robot.goal, 'x', color='yellow', markersize=10, markeredgewidth=2, zorder=1)
    if planner.resolved_goal is not None:
        ax.plot(*(np.asarray(planner.resolved_goal) + robot.pos), 'x', color='cyan',
                markersize=10, markeredgewidth=2, zorder=1)

    ring = lambda c, l: mlines.Line2D([], [], marker='o', linestyle='none', markersize=8, markerfacecolor='none',
                                      markeredgecolor=c, markeredgewidth=2, label=l)
    fill = lambda v, l: mpatches.Patch(color=VALENCE_CMAP(VALENCE_NORM(v)), label=l)
    legend = ([fill(-1, 'negative'), fill(0, 'neutral'), fill(1, 'positive')]
              + [ring(RESPONSE_COLORS[k], k) for k in ('attend', 'stop', 'sidestep', 'startle')]
              + [mpatches.Patch(color='red', alpha=0.3, label='personal space'),
                 mlines.Line2D([], [], color='red', linestyle=':', alpha=0.6, label='notice zone'),
                 mpatches.Patch(color='blue', alpha=0.3, label='predicted PS'),
                 mpatches.Patch(color='orange', label='robot')])
    ax.legend(handles=legend, loc='upper right', fontsize=6, framealpha=0.6, ncol=2)
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
