"""
tests/eval_speed_test.py — Core speed sweep runner

Called by run_test.py via run(). Can also be run standalone for quick testing:
    mjpython tests/eval_speed_test.py
"""

import mujoco
import mujoco.viewer
import numpy as np
import torch
import csv
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import ROBOT_DIR, POLICY_PATH

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
ANG_VEL_SCALE = 0.25    # scales angular velocity before feeding to policy
DOF_POS_SCALE = 1.0     # scales joint position offset from default
DOF_VEL_SCALE = 0.05    # scales joint velocity
ACTION_SCALE  = 0.25    # scales policy output back to joint space
CMD_SCALE     = np.array([2.0, 2.0, 0.25], dtype=np.float32)  # scales cmd input

NUM_ACTIONS = 12   # joints controlled
NUM_OBS     = 47   # total policy input dimensions

# ── Sweep config ──────────────────────────────────────────────────────────────
VX_SWEEP           = [round(v * 0.2, 1) for v in range(1, 16)]  # 0.2 → 3.0 m/s
EPISODES_PER_SPEED = 5
EPISODE_STEPS      = 5000   # 10 seconds per episode at dt=0.002
FALL_HEIGHT        = 0.4    # meters — torso below this = fall

# ── Log rate ──────────────────────────────────────────────────────────────────
LOG_EVERY_N_STEPS  = 100    # write one row every 100 physics steps (~0.2s)

# ── Direction → cmd vector ────────────────────────────────────────────────────
# vx values are always positive — direction determines which axis and sign
DIRECTION_MAP = {
    "forward":  lambda vx: np.array([ vx,  0, 0], dtype=np.float32),
    "backward": lambda vx: np.array([-vx,  0, 0], dtype=np.float32),
    "lateral":  lambda vx: np.array([  0, vx, 0], dtype=np.float32),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_gravity_orientation(quaternion):
    """Derive gravity vector in body frame from IMU quaternion.
    On flat ground returns ~[0, 0, -1].
    g[0] = forward tilt (pitch proxy)
    g[1] = lateral tilt (roll proxy)
    """
    qw, qx, qy, qz = quaternion
    g = np.zeros(3)
    g[0] =  2 * (-qz * qx + qw * qy)
    g[1] = -2 * ( qz * qy + qw * qx)
    g[2] =  1  - 2 * (qw * qw + qz * qz)
    return g

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD controller — computes torque commands for all 12 joints."""
    return (target_q - q) * kp + (target_dq - dq) * kd

def get_gait_phase(counter):
    """Returns (sin_phase, cos_phase) for current step counter.
    Period = 0.8s. sin+cos together encode full circular phase
    without discontinuity at the 0/1 boundary.
    """
    period = 0.8
    t      = counter * SIMULATION_DT
    phase  = (t % period) / period
    return np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)


# ── CSV logging ───────────────────────────────────────────────────────────────

def init_csv(csv_path):
    """Write header row. Called once before sweep starts."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "episode_id",    # which run
            "step",          # physics step within episode
            "commanded_vx",  # speed sent to policy (always positive)
            "pitch",         # gravity[0] — forward tilt, from IMU quaternion
            "roll",          # gravity[1] — lateral tilt, from IMU quaternion
            "omega_x",       # data.qvel[3] — roll rate, from IMU
            "omega_y",       # data.qvel[4] — pitch rate, from IMU
            "sin_phase",     # gait phase sine — where in step cycle
            "cos_phase",     # gait phase cosine — paired with sin
            "fall_detected", # True only at fall timestep, False otherwise
        ])

def log_row(csv_path, episode_id, step, vx,
            pitch, roll, omega_x, omega_y,
            sin_phase, cos_phase, fell):
    """Append one timestep row to CSV."""
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow([
            episode_id, step, vx,
            round(float(pitch),    4),
            round(float(roll),     4),
            round(float(omega_x),  4),
            round(float(omega_y),  4),
            round(float(sin_phase),4),
            round(float(cos_phase),4),
            fell,
        ])

def log_current_state(csv_path, episode_id, step, vx, data, counter, fell):
    """Compute all log fields from current sim state and write row.
    All fields are IMU/encoder-derived — real-robot-safe.
    torso height (data.qpos[2]) intentionally excluded — MuJoCo only.
    """
    gravity              = get_gravity_orientation(data.qpos[3:7])
    sin_phase, cos_phase = get_gait_phase(counter)
    log_row(
        csv_path, episode_id, step, vx,
        pitch     = gravity[0],
        roll      = gravity[1],
        omega_x   = data.qvel[3],
        omega_y   = data.qvel[4],
        sin_phase = sin_phase,
        cos_phase = cos_phase,
        fell      = fell,
    )


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(model, data, policy, vx, run_id, csv_path,
                cmd_fn, viewer_handle=None):
    """Run one episode. Returns (outcome, survival_time_s)."""
    mujoco.mj_resetData(model, data)

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

        # ── Periodic log ───────────────────────────────────────────────────
        if step % LOG_EVERY_N_STEPS == 0:
            log_current_state(csv_path, run_id, step, vx, data, counter, False)

        # ── Fall check ─────────────────────────────────────────────────────
        # torso height used only as trigger — not logged (MuJoCo-only state)
        if data.qpos[2] < FALL_HEIGHT:
            log_current_state(csv_path, run_id, step, vx, data, counter, True)
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

            sin_phase, cos_phase = get_gait_phase(counter)

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

    # ── Survived full episode ──────────────────────────────────────────────
    log_current_state(csv_path, run_id, EPISODE_STEPS - 1, vx, data, counter, False)
    return "timeout", EPISODE_STEPS * SIMULATION_DT


# ── Main sweep ────────────────────────────────────────────────────────────────

def run(scene_path, direction, csv_path, viewer=False):
    """Entry point called by run_test.py."""

    cmd_fn = DIRECTION_MAP[direction]

    # Load model
    original_dir = os.getcwd()
    os.chdir(os.path.dirname(scene_path))
    model = mujoco.MjModel.from_xml_path(os.path.basename(scene_path))
    model.opt.timestep = SIMULATION_DT
    os.chdir(original_dir)

    data   = mujoco.MjData(model)
    policy = torch.jit.load(POLICY_PATH)

    init_csv(csv_path)

    print(f"Logging to: {csv_path}")
    print(f"Speeds: {VX_SWEEP}")
    print(f"Episodes per speed: {EPISODES_PER_SPEED}\n")

    # ── Viewer or headless ────────────────────────────────────────────────
    def _sweep(viewer_handle=None):
        run_id = 0
        for vx in VX_SWEEP:
            print(f"── vx = {vx} m/s ──")
            falls = 0
            for ep in range(EPISODES_PER_SPEED):
                outcome, t = run_episode(
                    model, data, policy, vx, run_id,
                    csv_path, cmd_fn, viewer_handle
                )
                status = "FALL" if outcome == "fall" else "OK"
                print(f"  ep {ep+1}: {status} at {t:.2f}s")
                if outcome == "fall":
                    falls += 1
                run_id += 1
            print(f"  fall rate: {falls}/{EPISODES_PER_SPEED}\n")
        print(f"Done. Results → {csv_path}")

    if viewer:
        with mujoco.viewer.launch_passive(model, data) as v:
            _sweep(viewer_handle=v)
    else:
        _sweep()


# ── Standalone entry point ────────────────────────────────────────────────────

def main():
    """Quick standalone run — flat terrain, forward, headless."""
    import os
    scene_path = os.path.join(ROBOT_DIR, "scene.xml")
    csv_path   = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "results", "standalone_forward.csv"
    )
    run(scene_path=scene_path, direction="forward",
        csv_path=csv_path, viewer=False)

if __name__ == "__main__":
    main()
