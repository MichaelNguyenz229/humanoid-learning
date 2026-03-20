"""
sandbox.py — Manual viewer for quick auditing

Edit the config block below, then run:
    mjpython sandbox.py
"""

import mujoco
import mujoco.viewer
import numpy as np
import torch
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import ROBOT_DIR, POLICY_PATH

# ── Edit these ────────────────────────────────────────────────────────────────

SCENE    = "flat"       # flat | stairs_low | stairs_mid | stairs_high |
                        # slope_low | slope_mid | slope_high |
                        # octave_low | octave_mid | octave_high

VX       = 2.8          # speed in m/s (always positive)

DIRECTION = "backward"   # forward | backward | lateral

# ─────────────────────────────────────────────────────────────────────────────

# ── Constants ─────────────────────────────────────────────────────────────────
SIMULATION_DT      = 0.002
CONTROL_DECIMATION = 10
KPS = np.array([100,100,100,150,40,40,100,100,100,150,40,40], dtype=np.float32)
KDS = np.array([2,2,2,4,2,2,2,2,2,4,2,2],                    dtype=np.float32)
DEFAULT_ANGLES = np.array(
    [-0.1,0.0,0.0,0.3,-0.2,0.0,-0.1,0.0,0.0,0.3,-0.2,0.0],
    dtype=np.float32
)
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
ACTION_SCALE  = 0.25
CMD_SCALE     = np.array([2.0, 2.0, 0.25], dtype=np.float32)
NUM_ACTIONS   = 12
NUM_OBS       = 47

DIRECTION_MAP = {
    "forward":  lambda vx: np.array([ vx,  0, 0], dtype=np.float32),
    "backward": lambda vx: np.array([-vx,  0, 0], dtype=np.float32),
    "lateral":  lambda vx: np.array([  0, vx, 0], dtype=np.float32),
}

def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    g = np.zeros(3)
    g[0] =  2 * (-qz * qx + qw * qy)
    g[1] = -2 * ( qz * qy + qw * qx)
    g[2] =  1  - 2 * (qw * qw + qz * qz)
    return g

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


# ── Load scene ────────────────────────────────────────────────────────────────
if SCENE == "flat":
    scene_path = os.path.join(ROBOT_DIR, "scene.xml")
else:
    scene_path = os.path.join(os.path.dirname(__file__), "scenes", f"{SCENE}.xml")

if not os.path.exists(scene_path):
    print(f"Error: scene not found at {scene_path}")
    sys.exit(1)

original_dir = os.getcwd()
os.chdir(os.path.dirname(scene_path))
model = mujoco.MjModel.from_xml_path(os.path.basename(scene_path))
model.opt.timestep = SIMULATION_DT
os.chdir(original_dir)

data   = mujoco.MjData(model)
policy = torch.jit.load(POLICY_PATH)

cmd = DIRECTION_MAP[DIRECTION](VX)

print(f"\n── Sandbox ───────────────────────────────────────")
print(f"   scene      {SCENE}")
print(f"   direction  {DIRECTION}")
print(f"   vx         {VX} m/s")
print(f"   cmd        {cmd}")
print(f"   press ESC to exit")
print(f"──────────────────────────────────────────────────\n")

# ── Sim loop ──────────────────────────────────────────────────────────────────
action         = np.zeros(NUM_ACTIONS, dtype=np.float32)
target_dof_pos = DEFAULT_ANGLES.copy()
obs            = np.zeros(NUM_OBS,     dtype=np.float32)
counter        = 0

with mujoco.viewer.launch_passive(model, data) as viewer:

    # Track robot with camera
    viewer.cam.type        = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = model.body('pelvis').id
    viewer.cam.distance    = 3.0   # meters from robot 
    start = time.time()

    while viewer.is_running():
        step_start = time.time()

        # PD control
        tau = pd_control(target_dof_pos, data.qpos[7:], KPS,
                         np.zeros_like(KDS), data.qvel[6:], KDS)
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        counter += 1

        # Policy inference
        if counter % CONTROL_DECIMATION == 0:
            qj    = data.qpos[7:]
            dqj   = data.qvel[6:]
            quat  = data.qpos[3:7]
            omega = data.qvel[3:6]

            gravity    = get_gravity_orientation(quat)
            omega_sc   = omega * ANG_VEL_SCALE
            qj_scaled  = (qj - DEFAULT_ANGLES) * DOF_POS_SCALE
            dqj_scaled = dqj * DOF_VEL_SCALE

            period    = 0.8
            t         = counter * SIMULATION_DT
            phase     = (t % period) / period
            sin_phase = np.sin(2 * np.pi * phase)
            cos_phase = np.cos(2 * np.pi * phase)

            obs[:3]                                = omega_sc
            obs[3:6]                               = gravity
            obs[6:9]                               = cmd * CMD_SCALE
            obs[9:9+NUM_ACTIONS]                   = qj_scaled
            obs[9+NUM_ACTIONS:9+2*NUM_ACTIONS]     = dqj_scaled
            obs[9+2*NUM_ACTIONS:9+3*NUM_ACTIONS]   = action
            obs[9+3*NUM_ACTIONS:9+3*NUM_ACTIONS+2] = [sin_phase, cos_phase]

            obs_tensor     = torch.from_numpy(obs).unsqueeze(0)
            action         = policy(obs_tensor).detach().numpy().squeeze()
            target_dof_pos = action * ACTION_SCALE + DEFAULT_ANGLES

        viewer.sync()

        # Print state every 2 seconds
        if counter % 1000 == 0:
            elapsed = time.time() - start
            gravity = get_gravity_orientation(data.qpos[3:7])
            print(
                f"t={elapsed:.1f}s  "
                f"x={data.qpos[0]:.2f}m  "
                f"z={data.qpos[2]:.2f}m  "
                f"pitch={gravity[0]:.3f}  "
                f"roll={gravity[1]:.3f}"
            )

        # Real-time pacing
        elapsed_step = time.time() - step_start
        remaining    = model.opt.timestep - elapsed_step
        if remaining > 0:
            time.sleep(remaining)
