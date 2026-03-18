import mujoco
import mujoco.viewer
import os
import numpy as np
import torch
import time

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROBOT_DIR, POLICY_PATH

def create_staircase_xml(step_height,width,depth):

  stairs_xml = f'''
      <!-- Stairs -->
      <geom name="step1" type="box" pos="{1 + (depth*1)} 0 {step_height/2}" size="{depth/2} {width/2} {step_height/2}" rgba="0.6 0.4 0.2 1"/>
      <geom name="step2" type="box" pos="{1 + (depth*2)} 0 {step_height*2/2}" size="{depth/2} {width/2}  {step_height*2/2}" rgba="0.6 0.4 0.2 1"/>
      <geom name="step3" type="box" pos="{1 + (depth*3)} 0 {step_height*3/2}" size="{depth/2} {width/2} {step_height*3/2}" rgba="0.6 0.4 0.2 1"/>
      <geom name="step4" type="box" pos="{1 + (depth*4)} 0 {step_height*4/2}" size="{depth/2} {width/2} {step_height*4/2}" rgba="0.6 0.4 0.2 1"/>
      <geom name="step5" type="box" pos="{1 + (depth*5)} 0 {step_height*5/2}" size="{depth/2} {width/2} {step_height*5/2}" rgba="0.6 0.4 0.2 1"/>
    </worldbody>'''
  
  return stairs_xml

# Config (from g1.yaml)
SIMULATION_DT = 0.002 #Delta Time -> a physics step every 0.002 seconds -> 500 steps per second
CONTROL_DECIMATION = 10 # Policy runs every 10 physics steps
KPS = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40], dtype=np.float32) #KP for each 12 joints
KDS = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2], dtype=np.float32) #KD for each 12 joints
DEFAULT_ANGLES = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0], dtype=np.float32) #Default joint angles
ANG_VEL_SCALE = 0.25 #Configured angular velocity scaler for specific policy
DOF_POS_SCALE = 1.0 #Configured DOF position scaler for specific policy
DOF_VEL_SCALE = 0.05 #DOF velocity scalaer for specific policy
ACTION_SCALE = 0.25 #Action scaler means previous actions - this scales the actions output
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32) #CMD scaler for policy
NUM_ACTIONS = 12 #Total number of actions (joint control)
NUM_OBS = 47 #Total number of observations (angular velocity, gravity orientation, command, joint positions, joint velocities, previous actions, gait phase)
CMD = np.array([2, 0, 0], dtype=np.float32)  # Walk forward slowly


def get_gravity_orientation(quaternion):
    """Calculate gravity vector in body frame"""
    qw, qx, qy, qz = quaternion #This is mujoco's convention, splitting this insto 4 variable from data.qpos[3:7]
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





#stairs_xml = create_staircase_xml(step_height=0.225, width=1.0, depth=0.6)  # 50% taller - how much sooner does it fail?
#stairs_xml = create_staircase_xml(step_height=0.15, width=1.0, depth=0.6)   # baseline
stairs_xml = create_staircase_xml(step_height=0.075, width=1.0, depth=0.6)  # 50% shorter - can it go longer?






modified_xml = xml_content.replace('</worldbody>', stairs_xml)


print("Loading G1 with stairs...")
model = mujoco.MjModel.from_xml_path("scene.xml")
model = mujoco.MjModel.from_xml_string(modified_xml)
#model = mujoco.MjModel.from_xml_string(xml_content)
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
            
            if counter == CONTROL_DECIMATION:
                print("=== OBS VECTOR AT STEP 0 ===")
                print(obs)
                print(f"obs[0]: {obs[0]}")
                print(f"obs[3]: {obs[3]}")
                print(f"obs[6]: {obs[6]}")
                print(f"obs[9]: {obs[9]}")
                print(f"obs[45]: {obs[45]}")
            
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
        if counter % 100 == 0:
            elapsed = time.time() - start_time
            x_pos = data.qpos[0]
            print(f"Time: {elapsed:.1f}s | X: {x_pos:.2f}m | Z: {data.qpos[2]:.2f}m")

print("\n✅ Simulation complete!")