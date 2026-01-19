import mujoco
import mujoco.viewer
import os
import numpy as np
import torch
import time

# Paths
PROJECT_ROOT = os.path.abspath("..")
ROBOT_DIR = os.path.join(PROJECT_ROOT, "models", "unitree_rl_gym", "resources", "robots", "g1_description")
POLICY_PATH = os.path.join(PROJECT_ROOT, "models", "unitree_rl_gym", "deploy", "pre_train", "g1", "motion.pt")

# Config (from g1.yaml)
SIMULATION_DT = 0.002
CONTROL_DECIMATION = 10
KPS = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40], dtype=np.float32)
KDS = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2], dtype=np.float32)
DEFAULT_ANGLES = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0], dtype=np.float32)
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
ACTION_SCALE = 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
NUM_ACTIONS = 12
NUM_OBS = 47
CMD = np.array([0.5, 0, 0], dtype=np.float32)  # Walk forward slowly

def get_gravity_orientation(quaternion):
    """Calculate gravity vector in body frame"""
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD controller for joint control"""
    return (target_q - q) * kp + (target_dq - dq) * kd

# Change to robot directory
original_dir = os.getcwd()
os.chdir(ROBOT_DIR)

# Load scene and add stairs
with open("scene.xml", 'r') as f:
    xml_content = f.read()

stairs_xml = '''
    <!-- Stairs -->
    <geom name="step1" type="box" pos="1.0 0 0.075" size="0.3 0.5 0.075" rgba="0.6 0.4 0.2 1"/>
    <geom name="step2" type="box" pos="1.6 0 0.225" size="0.3 0.5 0.075" rgba="0.6 0.4 0.2 1"/>
    <geom name="step3" type="box" pos="2.2 0 0.375" size="0.3 0.5 0.075" rgba="0.6 0.4 0.2 1"/>
    <geom name="step4" type="box" pos="2.8 0 0.525" size="0.3 0.5 0.075" rgba="0.6 0.4 0.2 1"/>
    <geom name="step5" type="box" pos="3.4 0 0.675" size="0.3 0.5 0.075" rgba="0.6 0.4 0.2 1"/>
  </worldbody>'''

modified_xml = xml_content.replace('</worldbody>', stairs_xml)

print("Loading G1 with stairs...")
model = mujoco.MjModel.from_xml_path("scene.xml")
model = mujoco.MjModel.from_xml_string(modified_xml)
data = mujoco.MjData(model)
model.opt.timestep = SIMULATION_DT

# Load policy
print("Loading pre-trained walking policy...")
policy = torch.jit.load(POLICY_PATH)
print("✅ Policy loaded!")

# Change back to original directory
os.chdir(original_dir)

# Initialize variables
action = np.zeros(NUM_ACTIONS, dtype=np.float32)
target_dof_pos = DEFAULT_ANGLES.copy()
obs = np.zeros(NUM_OBS, dtype=np.float32)
counter = 0

print(f"\n🤖 Starting simulation!")
print(f"   - Command: Walk forward at {CMD[0]} m/s")
print(f"   - Press ESC to exit\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    
    while viewer.is_running():
        step_start = time.time()
        
        # Apply PD control
        tau = pd_control(target_dof_pos, data.qpos[7:], KPS, np.zeros_like(KDS), data.qvel[6:], KDS)
        data.ctrl[:] = tau
        
        # Step physics
        mujoco.mj_step(model, data)
        counter += 1
        
        # Policy inference at lower frequency
        if counter % CONTROL_DECIMATION == 0:
            # Get robot state
            qj = data.qpos[7:]  # Joint positions
            dqj = data.qvel[6:]  # Joint velocities
            quat = data.qpos[3:7]  # Body orientation
            omega = data.qvel[3:6]  # Body angular velocity
            
            # Scale observations
            qj_scaled = (qj - DEFAULT_ANGLES) * DOF_POS_SCALE
            dqj_scaled = dqj * DOF_VEL_SCALE
            gravity_orientation = get_gravity_orientation(quat)
            omega_scaled = omega * ANG_VEL_SCALE
            
            # Gait phase (periodic signal for leg coordination)
            period = 0.8
            count = counter * SIMULATION_DT
            phase = (count % period) / period
            sin_phase = np.sin(2 * np.pi * phase)
            cos_phase = np.cos(2 * np.pi * phase)
            
            # Build observation vector
            obs[:3] = omega_scaled
            obs[3:6] = gravity_orientation
            obs[6:9] = CMD * CMD_SCALE
            obs[9:9+NUM_ACTIONS] = qj_scaled
            obs[9+NUM_ACTIONS:9+2*NUM_ACTIONS] = dqj_scaled
            obs[9+2*NUM_ACTIONS:9+3*NUM_ACTIONS] = action
            obs[9+3*NUM_ACTIONS:9+3*NUM_ACTIONS+2] = np.array([sin_phase, cos_phase])
            
            # Policy inference
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            action = policy(obs_tensor).detach().numpy().squeeze()
            
            # Convert action to target joint positions
            target_dof_pos = action * ACTION_SCALE + DEFAULT_ANGLES
        
        # Sync viewer
        viewer.sync()
        
        # Time keeping
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        
        # Print progress every 2 seconds
        if counter % 1000 == 0:
            elapsed = time.time() - start_time
            x_pos = data.qpos[0]
            print(f"Time: {elapsed:.1f}s | Robot X position: {x_pos:.2f}m")

print("\n✅ Simulation complete!")