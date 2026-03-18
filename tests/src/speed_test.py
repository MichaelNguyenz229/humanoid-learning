# tests/eval_speed_test.py

import mujoco
import numpy as np
import torch
import csv
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import ROBOT_DIR, POLICY_PATH

# ── Constants ─────────────────────────────────────────────────────────────────
SIMULATION_DT      = 0.002
CONTROL_DECIMATION = 10
KPS = np.array([100,100,100,150,40,40,100,100,100,150,40,40], dtype=np.float32)
KDS = np.array([2,2,2,4,2,2,2,2,2,4,2,2], dtype=np.float32)
DEFAULT_ANGLES = np.array([-0.1,0.0,0.0,0.3,-0.2,0.0,-0.1,0.0,0.0,0.3,-0.2,0.0], dtype=np.float32)
ANG_VEL_SCALE  = 0.25
DOF_POS_SCALE  = 1.0
DOF_VEL_SCALE  = 0.05
ACTION_SCALE   = 0.25
CMD_SCALE      = np.array([2.0, 2.0, 0.25], dtype=np.float32)
NUM_ACTIONS    = 12
NUM_OBS        = 47

# ── Sweep config ──────────────────────────────────────────────────────────────
VX_SWEEP           = [round(v * 0.2, 1) for v in range(1, 16)]  # 0.2 → 3.0
EPISODES_PER_SPEED = 5
EPISODE_STEPS      = 5000  # 10 seconds at 0.002 dt
FALL_HEIGHT        = 0.4   # meters

# ── Output ────────────────────────────────────────────────────────────────────
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "speed_test.csv")


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    g = np.zeros(3)
    g[0] =  2 * (-qz * qx + qw * qy)
    g[1] = -2 * ( qz * qy + qw * qx)
    g[2] =  1  - 2 * (qw * qw + qz * qz)
    return g

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

def init_csv():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow([
            "episode_id", "step", "commanded_vx",
            "pitch", "mean_torque", "torso_height", "fall_detected"
        ])

def log_timestep(episode_id, step, vx, pitch, mean_torque, torso_height, fell):
    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([
            episode_id, step, vx,
            round(float(pitch), 4),
            round(float(mean_torque), 4),
            round(float(torso_height), 4),
            fell
        ])

def run_episode(model, data, policy, vx, run_id, episode):
    """Run one episode headlessly. Returns (outcome, survival_time_s)."""
    mujoco.mj_resetData(model, data)

    cmd            = np.array([vx, 0, 0], dtype=np.float32)
    action         = np.zeros(NUM_ACTIONS, dtype=np.float32)
    target_dof_pos = DEFAULT_ANGLES.copy()
    obs            = np.zeros(NUM_OBS, dtype=np.float32)
    tau            = np.zeros(NUM_ACTIONS, dtype=np.float32)  # init so step-0 log is safe
    counter        = 0

    for step in range(EPISODE_STEPS):

        # ── PD control + physics step ──────────────────────────────────────
        tau = pd_control(target_dof_pos, data.qpos[7:], KPS,
                         np.zeros_like(KDS), data.qvel[6:], KDS)
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        counter += 1

        # ── Timestep log every 50 steps ───────────────────────────────────
        if step % 50 == 0:
            gravity = get_gravity_orientation(data.qpos[3:7])
            log_timestep(
                episode_id  = run_id,
                step        = step,
                vx          = vx,
                pitch       = gravity[0],
                mean_torque = np.mean(np.abs(tau)),
                torso_height= data.qpos[2],
                fell        = False
            )

        # ── Fall check ────────────────────────────────────────────────────
        if data.qpos[2] < FALL_HEIGHT:
            survival_time = step * SIMULATION_DT
            gravity = get_gravity_orientation(data.qpos[3:7])
            log_timestep(
                episode_id  = run_id,
                step        = step,
                vx          = vx,
                pitch       = gravity[0],
                mean_torque = np.mean(np.abs(tau)),
                torso_height= data.qpos[2],
                fell        = True
            )
            return "fall", survival_time

        # ── Policy inference ──────────────────────────────────────────────
        if counter % CONTROL_DECIMATION == 0:
            qj    = data.qpos[7:]
            dqj   = data.qvel[6:]
            quat  = data.qpos[3:7]
            omega = data.qvel[3:6]

            qj_scaled  = (qj - DEFAULT_ANGLES) * DOF_POS_SCALE
            dqj_scaled = dqj * DOF_VEL_SCALE
            gravity    = get_gravity_orientation(quat)
            omega_sc   = omega * ANG_VEL_SCALE

            period = 0.8
            t      = counter * SIMULATION_DT
            phase  = (t % period) / period

            obs[:3]                                    = omega_sc
            obs[3:6]                                   = gravity
            obs[6:9]                                   = cmd * CMD_SCALE
            obs[9:9+NUM_ACTIONS]                       = qj_scaled
            obs[9+NUM_ACTIONS:9+2*NUM_ACTIONS]         = dqj_scaled
            obs[9+2*NUM_ACTIONS:9+3*NUM_ACTIONS]       = action
            obs[9+3*NUM_ACTIONS:9+3*NUM_ACTIONS+2]     = [
                np.sin(2 * np.pi * phase),
                np.cos(2 * np.pi * phase)
            ]

            obs_tensor     = torch.from_numpy(obs).unsqueeze(0)
            action         = policy(obs_tensor).detach().numpy().squeeze()
            target_dof_pos = action * ACTION_SCALE + DEFAULT_ANGLES

    # ── Survived full episode ─────────────────────────────────────────────
    gravity = get_gravity_orientation(data.qpos[3:7])
    log_timestep(
        episode_id  = run_id,
        step        = EPISODE_STEPS - 1,
        vx          = vx,
        pitch       = gravity[0],
        mean_torque = np.mean(np.abs(tau)),
        torso_height= data.qpos[2],
        fell        = False
    )
    return "timeout", EPISODE_STEPS * SIMULATION_DT


def main():
    original_dir = os.getcwd()
    os.chdir(ROBOT_DIR)
    model = mujoco.MjModel.from_xml_path("scene.xml")
    model.opt.timestep = SIMULATION_DT
    os.chdir(original_dir)

    data   = mujoco.MjData(model)
    policy = torch.jit.load(POLICY_PATH)

    init_csv()

    run_id = 0
    for vx in VX_SWEEP:
        print(f"\n── vx = {vx} m/s ──")
        falls = 0
        for ep in range(EPISODES_PER_SPEED):
            outcome, t = run_episode(model, data, policy, vx, run_id, ep + 1)
            status = "FALL" if outcome == "fall" else "OK"
            print(f"  ep {ep+1}: {status} at {t:.2f}s")
            if outcome == "fall":
                falls += 1
            run_id += 1
        print(f"  fall rate: {falls}/{EPISODES_PER_SPEED}")

    print(f"\nDone. Results → {LOG_PATH}")

if __name__ == "__main__":
    main()