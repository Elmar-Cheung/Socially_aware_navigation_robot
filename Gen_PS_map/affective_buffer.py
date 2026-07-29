# affective_buffer.py
import numpy as np

AFFECT_BUFFER = {
    "observations": [],   # raw obs dicts from env
    "actions":      [],   # actions taken
    "valence":      [],   # human emotion valence
    "intensity":      [],   # human emotion intensity
    "distances":    [],   # robot-to-human distances
    "timesteps":    []    # when it was collected
}

def collect(obs, action, humans, timestep):
    
    # extract affect from nearest human
    if humans:
        nearest = min(humans, key=lambda h: np.linalg.norm(h.pos))
        valence = nearest.emotion[0]
        intensity = nearest.emotion[1]
        dist    = float(np.linalg.norm(nearest.pos))
    else:
        valence, intensity, dist = 0.0, 0.0, 999.0

    AFFECT_BUFFER["observations"].append(obs)
    AFFECT_BUFFER["actions"].append(action)
    AFFECT_BUFFER["valence"].append(valence)
    AFFECT_BUFFER["intensity"].append(intensity)
    AFFECT_BUFFER["distances"].append(dist)
    AFFECT_BUFFER["timesteps"].append(timestep)

def get_buffer():
    return AFFECT_BUFFER

def clear():
    for key in AFFECT_BUFFER:
        AFFECT_BUFFER[key].clear()