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
EPISODE_STEPS      = 5000  # 10 seconds at dt=0.002
FALL_HEIGHT        = 0.4   # meters — robot considered fallen below this torso height

# ── Output ────────────────────────────────────────────────────────────────────
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "collections_data", "fall_data.csv")

# ── Logging fields (all real-robot-safe — no MuJoCo-only state) ───────────────
#
#   episode_id    — which run (groups timesteps into episodes)
#   step          — timestep within episode
#   commanded_vx  — forward speed sent to policy (you control this)
#
#   pitch         — gravity[0]: forward tilt of torso, derived from IMU quaternion
#   roll          — gravity[1]: lateral tilt of torso, derived from IMU quaternion
#   omega_x       — data.qvel[3]: roll rate from IMU angular velocity
#   omega_y       — data.qvel[4]: pitch rate from IMU angular velocity
#
#   sin_phase     — sin(2π * gait_phase): where in step cycle the robot is
#   cos_phase     — cos(2π * gait_phase): paired with sin to encode full phase
#                   sin+cos together encode full circular phase without discontinuity
#
#   mean_torque   — average |tau| across all 12 joints: proxy for motor current load
#                   high torque = robot fighting terrain or destabilizing
#                   available on real robot via motor current sensors
#
#   fall_detected — True only at the timestep a fall is confirmed (torso < FALL_HEIGHT)
#                   all other rows are False
#                   friend adds fall_within_2s label in postprocessing


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    g = np.zeros(3)
    g[0] =  2 * (-qz * qx + qw * qy)   # forward tilt (pitch proxy)
    g[1] = -2 * ( qz * qy + qw * qx)   # lateral tilt (roll proxy)
    g[2] =  1  - 2 * (qw * qw + qz * qz)
    return g

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

def init_csv():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow([
            "episode_id", "step", "commanded_vx",
            "pitch", "roll",
            "omega_x", "omega_y",
            "sin_phase", "cos_phase",
            "mean_torque",
            "fall_detected"
        ])

def log_timestep(episode_id, step, vx,
                 pitch, roll, omega_x, omega_y,
                 sin_phase, cos_phase,
                 mean_torque, fell):
    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([
            episode_id, step, vx,
            round(float(pitch), 4),
            round(float(roll), 4),
            round(float(omega_x), 4),
            round(float(omega_y), 4),
            round(float(sin_phase), 4),
            round(float(cos_phase), 4),
            round(float(mean_torque), 4),
            fell
        ])

def compute_log_fields(data, tau, counter):
    """Compute all loggable fields from current sim state.
    All fields derived from IMU, encoders, or PD output — real-robot-safe.
    Nothing from data.qpos[0:3] (position) or data.qvel[0:3] (linear vel) —
    those are MuJoCo-only and not available on real hardware.
    """
    gravity = get_gravity_orientation(data.qpos[3:7])  # orientation from IMU quaternion

    pitch   = gravity[0]   # forward tilt — IMU
    roll    = gravity[1]   # lateral tilt — IMU
    omega_x = data.qvel[3] # roll rate    — IMU
    omega_y = data.qvel[4] # pitch rate   — IMU

    # Gait phase — encodes where in the step cycle the robot currently is
    # sin+cos used together so phase wraps smoothly (no discontinuity at 0/1 boundary)
    period    = 0.8
    t         = counter * SIMULATION_DT
    phase     = (t % period) / period
    sin_phase = np.sin(2 * np.pi * phase)
    cos_phase = np.cos(2 * np.pi * phase)

    # Mean absolute torque across all 12 joints
    # PD controller output — proxy for motor current load on real hardware
    # spikes when joints deviate far from target (fighting terrain / destabilizing)
    mean_torque = np.mean(np.abs(tau))

    return pitch, roll, omega_x, omega_y, sin_phase, cos_phase, mean_torque


def run_episode(model, data, policy, vx, run_id, episode):
    """Run one episode headlessly. Returns (outcome, survival_time_s)."""
    mujoco.mj_resetData(model, data)

    cmd            = np.array([vx, 0, 0], dtype=np.float32)
    action         = np.zeros(NUM_ACTIONS, dtype=np.float32)
    target_dof_pos = DEFAULT_ANGLES.copy()
    obs            = np.zeros(NUM_OBS, dtype=np.float32)
    tau            = np.zeros(NUM_ACTIONS, dtype=np.float32)  # init before loop so step-0 log is safe
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
            pitch, roll, omega_x, omega_y, sin_phase, cos_phase, mean_torque = \
                compute_log_fields(data, tau, counter)
            log_timestep(
                episode_id=run_id, step=step, vx=vx,
                pitch=pitch, roll=roll,
                omega_x=omega_x, omega_y=omega_y,
                sin_phase=sin_phase, cos_phase=cos_phase,
                mean_torque=mean_torque,
                fell=False
            )

        # ── Fall check ────────────────────────────────────────────────────
        # data.qpos[2] (torso height) used only as fall trigger — NOT logged
        # it is MuJoCo-only state, not available on real hardware
        if data.qpos[2] < FALL_HEIGHT:
            survival_time = step * SIMULATION_DT
            pitch, roll, omega_x, omega_y, sin_phase, cos_phase, mean_torque = \
                compute_log_fields(data, tau, counter)
            log_timestep(
                episode_id=run_id, step=step, vx=vx,
                pitch=pitch, roll=roll,
                omega_x=omega_x, omega_y=omega_y,
                sin_phase=sin_phase, cos_phase=cos_phase,
                mean_torque=mean_torque,
                fell=True  # only True row in this episode
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

            obs[:3]                                = omega_sc
            obs[3:6]                               = gravity
            obs[6:9]                               = cmd * CMD_SCALE
            obs[9:9+NUM_ACTIONS]                   = qj_scaled
            obs[9+NUM_ACTIONS:9+2*NUM_ACTIONS]     = dqj_scaled
            obs[9+2*NUM_ACTIONS:9+3*NUM_ACTIONS]   = action
            obs[9+3*NUM_ACTIONS:9+3*NUM_ACTIONS+2] = [
                np.sin(2 * np.pi * phase),
                np.cos(2 * np.pi * phase)
            ]

            obs_tensor     = torch.from_numpy(obs).unsqueeze(0)
            action         = policy(obs_tensor).detach().numpy().squeeze()
            target_dof_pos = action * ACTION_SCALE + DEFAULT_ANGLES

    # ── Survived full episode ─────────────────────────────────────────────
    pitch, roll, omega_x, omega_y, sin_phase, cos_phase, mean_torque = \
        compute_log_fields(data, tau, counter)
    log_timestep(
        episode_id=run_id, step=EPISODE_STEPS - 1, vx=vx,
        pitch=pitch, roll=roll,
        omega_x=omega_x, omega_y=omega_y,
        sin_phase=sin_phase, cos_phase=cos_phase,
        mean_torque=mean_torque,
        fell=False
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