"""
eval_speed_test.py — Speed sweep runner, episode-level logging only

Called by run_test.py:
    mjpython run_test.py --scene flat --direction forward

Standalone:
    mjpython eval_speed_test.py
"""

import mujoco
import mujoco.viewer
import numpy as np
import torch
import csv
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import POLICY_PATH, SCENE_DIR

# ── Physics constants ─────────────────────────────────────────────────────────
SIMULATION_DT      = 0.002   # physics step size — 500 steps/sec
CONTROL_DECIMATION = 10      # policy runs every 10 physics steps = 50Hz

# ── Robot constants (from g1.yaml) ────────────────────────────────────────────
KPS = np.array([100,100,100,150,40,40,100,100,100,150,40,40], dtype=np.float32)
KDS = np.array([2,2,2,4,2,2,2,2,2,4,2,2],                    dtype=np.float32)
DEFAULT_ANGLES = np.array(
    [-0.1,0.0,0.0,0.3,-0.2,0.0,-0.1,0.0,0.0,0.3,-0.2,0.0],
    dtype=np.float32
)

# ── Observation scaling (from g1.yaml) ────────────────────────────────────────
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
ACTION_SCALE  = 0.25
CMD_SCALE     = np.array([2.0, 2.0, 0.25], dtype=np.float32)

NUM_ACTIONS = 12
NUM_OBS     = 47

# ── Sweep config ──────────────────────────────────────────────────────────────
VX_SWEEP           = [round(v * 0.2, 1) for v in range(1, 16)]  # 0.2 → 3.0 m/s
EPISODES_PER_SPEED = 5
EPISODE_STEPS      = 5000   # 10 seconds per episode at dt=0.002
FALL_HEIGHT        = 0.4    # meters — torso below this = fall

# ── Direction → cmd vector ────────────────────────────────────────────────────
DIRECTION_MAP = {
    "f":  lambda vx: np.array([ vx,  0, 0], dtype=np.float32),
    "b": lambda vx: np.array([-vx,  0, 0], dtype=np.float32),
    "l":  lambda vx: np.array([  0, vx, 0], dtype=np.float32),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    g = np.zeros(3)
    g[0] =  2 * (-qz * qx + qw * qy)
    g[1] = -2 * ( qz * qy + qw * qx)
    g[2] =  1  - 2 * (qw * qw + qz * qz)
    return g

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


# ── CSV logging — one row per episode ─────────────────────────────────────────

def init_csv(csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "episode_id",      # unique episode index across full sweep
            "commanded_vx",    # speed tested (always positive)
            "direction",       # forward | backward | lateral
            "outcome",         # fall | timeout
            "survival_time_s", # seconds survived (10.0 = full episode)
        ])

def log_episode(csv_path, episode_id, vx, direction, outcome, survival_time):
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow([
            episode_id,
            vx,
            direction,
            outcome,
            round(float(survival_time), 3),
        ])


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(model, data, policy, vx, cmd_fn, initial_quat=None, viewer_handle=None):
    mujoco.mj_resetData(model, data)

    # Set initial orientation if specified
    if initial_quat is not None:
        data.qpos[3:7] = initial_quat

    mujoco.mj_forward(model, data)  # recompute derived quantities after manual qpos set

    cmd            = cmd_fn(vx)
    action         = np.zeros(NUM_ACTIONS, dtype=np.float32)
    target_dof_pos = DEFAULT_ANGLES.copy()
    obs            = np.zeros(NUM_OBS,     dtype=np.float32)
    counter        = 0

    for step in range(EPISODE_STEPS):

        # ── PD control + physics ───────────────────────────────────────────
        tau = pd_control(target_dof_pos, data.qpos[7:], KPS,
                         np.zeros_like(KDS), data.qvel[6:], KDS)
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        counter += 1

        # ── Fall check ─────────────────────────────────────────────────────
        if data.qpos[2] < FALL_HEIGHT:
            return "fall", step * SIMULATION_DT

        # ── Policy inference ───────────────────────────────────────────────
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

        # ── Viewer sync ────────────────────────────────────────────────────
        if viewer_handle is not None:
            
            viewer_handle.sync()

    return "timeout", EPISODE_STEPS * SIMULATION_DT


# ── Main sweep ────────────────────────────────────────────────────────────────
QUAT_MAP = {
    "f":  None,
    "b": [0, 0, 0, 1],
    "l":  [0.7071, 0, 0, -0.7071],
}

def run(scene_path, direction, csv_path, viewer=False):
    """Entry point called by run_test.py."""
    cmd_fn = DIRECTION_MAP[direction]
    initial_quat = QUAT_MAP[direction]

    # Load model
    original_dir = os.getcwd()
    os.chdir(os.path.dirname(scene_path))
    model = mujoco.MjModel.from_xml_path(os.path.basename(scene_path))
    model.opt.timestep = SIMULATION_DT
    os.chdir(original_dir)

    data   = mujoco.MjData(model)
    policy = torch.jit.load(POLICY_PATH)

    init_csv(csv_path)

    print(f"Speeds:            {VX_SWEEP}")
    print(f"Episodes per speed: {EPISODES_PER_SPEED}")
    print(f"Logging to:         {csv_path}\n")

    def _sweep(viewer_handle=None):
        run_id = 0
        for vx in VX_SWEEP:
            print(f"── vx = {vx} m/s ──")
            falls = 0
            for ep in range(EPISODES_PER_SPEED):
                outcome, t = run_episode(
                    model, data, policy, vx, cmd_fn, initial_quat, viewer_handle
                )
                # Log one row per episode
                log_episode(csv_path, run_id, vx, direction, outcome, t)

                status = "FALL" if outcome == "fall" else "OK"
                print(f"  ep {ep+1}: {status} at {t:.2f}s")
                if outcome == "fall":
                    falls += 1
                run_id += 1
            print(f"  fall rate: {falls}/{EPISODES_PER_SPEED}\n")
        print(f"Done. Results → {csv_path}")

    if viewer:
        with mujoco.viewer.launch_passive(model, data) as v:
            v.cam.type        = mujoco.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = model.body('pelvis').id
            v.cam.distance    = 5
            v.cam.elevation   = -20
            _sweep(viewer_handle=v)
    else:
        _sweep()


# ── Standalone ────────────────────────────────────────────────────────────────

def main():
    scene_path = os.path.join(SCENE_DIR, "flat.xml")
    csv_path   = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", "standalone_forward.csv"
    )
    run(scene_path=scene_path, direction="f",
        csv_path=csv_path, viewer=False)

if __name__ == "__main__":
    main()
