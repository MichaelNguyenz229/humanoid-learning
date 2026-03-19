# Observation Space & Fall Detection Logging

## 47-Dimensional Observation Vector

The pretrained G1 policy takes a fixed 47-dim input vector built from onboard sensor state at each control step (every 10 physics steps, 50Hz).

| Indices | Field | Source | Scale |
|---------|-------|--------|-------|
| `[0:3]` | Angular velocity (roll, pitch, yaw rate) | IMU | `× 0.25` |
| `[3:6]` | Gravity orientation (pitch, roll, tilt) | IMU quaternion | — |
| `[6:9]` | Command velocity (vx, vy, yaw) | External cmd | `× [2.0, 2.0, 0.25]` |
| `[9:21]` | Joint positions (12 joints) | Encoders | `(q - default) × 1.0` |
| `[21:33]` | Joint velocities (12 joints) | Encoders | `× 0.05` |
| `[33:45]` | Previous actions (12 joints) | Internal | `× 0.25` |
| `[45:47]` | Gait phase `[sin, cos]` | Internal clock | `2π × phase` |

The gravity orientation vector `[g0, g1, g2]` is derived from the IMU quaternion via:

```python
g[0] =  2 * (-qz*qx + qw*qy)   # forward tilt
g[1] = -2 * ( qz*qy + qw*qx)   # lateral tilt
g[2] =  1  -  2*(qw*qw + qz*qz) # vertical alignment (~-1 on flat ground)
```

On flat ground `g ≈ [0, 0, -1]`. Deviations in `g[0]` and `g[1]` indicate tilt.

The gait phase uses `sin` and `cos` together so the signal wraps continuously without a discontinuity at the 0/1 boundary.

---

## Fall Detection Logging (`collections/collections_data/fall_data.csv`)

To support a data-driven fall predictor, `speed_collections.py` logs the following fields every 50 steps per episode. All fields are derived exclusively from onboard sensors — no MuJoCo-only state — making any model trained on this data transferable to real hardware.

| Field | Source | Description |
|-------|--------|-------------|
| `episode_id` | Internal | Groups timesteps into episodes |
| `step` | Internal | Timestep within episode |
| `commanded_vx` | External cmd | Forward speed sent to policy |
| `pitch` | IMU (`g[0]`) | Forward tilt of torso |
| `roll` | IMU (`g[1]`) | Lateral tilt of torso |
| `omega_x` | IMU (`qvel[3]`) | Roll rate |
| `omega_y` | IMU (`qvel[4]`) | Pitch rate |
| `sin_phase` | Internal clock | Gait phase sine component |
| `cos_phase` | Internal clock | Gait phase cosine component |
| `mean_torque` | PD output | Mean `\|tau\|` across 12 joints — proxy for motor load |
| `fall_detected` | Internal | `True` only at fall timestep — label for fall predictor |

### What is NOT logged and why

`torso_height` (`data.qpos[2]`) and root linear velocity (`data.qvel[0:3]`) are available in MuJoCo but not on the real robot. They are intentionally excluded so any model trained on this data is sim-to-real transferable. `torso_height` is still used internally as the fall trigger threshold but never written to the CSV.

### Fall label postprocessing

The `fall_detected` column marks only the exact fall timestep. A collaborator can derive a predictive label (`fall_within_2s`) in pandas:

```python
def label_falls(group):
    fall_steps = group[group['fall_detected'] == True]['step'].values
    if len(fall_steps) == 0:
        group['fall_within_2s'] = 0
    else:
        fall_step = fall_steps[0]
        group['fall_within_2s'] = (
            (group['step'] >= fall_step - 100) &
            (group['step'] <  fall_step)
        ).astype(int)
    return group

df = df.groupby('episode_id', group_keys=False).apply(label_falls)
```

This labels all timesteps within 100 steps (2 seconds) of a fall as positive examples for training.