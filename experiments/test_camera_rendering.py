import mujoco
import mujoco.viewer
import os
import numpy as np
import cv2
import torch
import time

# Paths
PROJECT_ROOT = os.path.abspath("..")
ROBOT_DIR = os.path.join(PROJECT_ROOT, "models", "unitree_rl_gym", "resources", "robots", "g1_description")
POLICY_PATH = os.path.join(PROJECT_ROOT, "models", "unitree_rl_gym", "deploy", "pre_train", "g1", "motion.pt")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "camera_views")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Config
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
CMD = np.array([2, 0, 0], dtype=np.float32)

def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

# Change to robot directory
original_dir = os.getcwd()
os.chdir(ROBOT_DIR)

# Load scene and add FUN ENVIRONMENT!
with open("scene.xml", 'r') as f:
    xml_content = f.read()

fun_scene_xml = '''
    <!-- Colorful boxes closer to path (left side) -->
    <geom name="box1" type="box" pos="3 -1.2 0.15" size="0.3 0.3 0.3" rgba="1 0.2 0.2 1"/>
    <geom name="box2" type="box" pos="5 -1.5 0.2" size="0.4 0.4 0.4" rgba="0.2 1 0.2 1"/>
    <geom name="box3" type="box" pos="7 -1.3 0.25" size="0.5 0.5 0.5" rgba="0.2 0.2 1 1"/>
    
    <!-- Colorful boxes (right side) -->
    <geom name="box4" type="box" pos="4 1.2 0.15" size="0.3 0.3 0.3" rgba="1 1 0.2 1"/>
    <geom name="box5" type="box" pos="6 1.4 0.2" size="0.35 0.35 0.35" rgba="1 0.5 0 1"/>
    <geom name="box6" type="box" pos="8 1.3 0.18" size="0.4 0.4 0.4" rgba="0.5 0 1 1"/>
    
    <!-- Colorful spheres (left side) -->
    <geom name="ball1" type="sphere" pos="2 -1.0 0.2" size="0.2" rgba="1 0.5 0 1"/>
    <geom name="ball2" type="sphere" pos="6 -1.6 0.25" size="0.25" rgba="0 1 1 1"/>
    
    <!-- Colorful spheres (right side) -->
    <geom name="ball3" type="sphere" pos="4 1.5 0.15" size="0.15" rgba="0.5 0 1 1"/>
    <geom name="ball4" type="sphere" pos="8 1.2 0.18" size="0.18" rgba="1 0 0.5 1"/>
    
    <!-- Cylinders (like pillars) -->
    <geom name="pillar1" type="cylinder" pos="5 -1.8 0.5" size="0.15 0.5" rgba="0.8 0.4 0 1"/>
    <geom name="pillar2" type="cylinder" pos="7 1.6 0.6" size="0.2 0.6" rgba="0.4 0 0.8 1"/>
    
    <!-- Small ramps on sides -->
    <geom name="ramp_left" type="box" pos="3 -1.5 0.1" size="0.8 0.3 0.01" 
          euler="0 0.3 0" rgba="0.6 0.6 0.3 1"/>
    <geom name="ramp_right" type="box" pos="9 1.5 0.1" size="0.8 0.3 0.01" 
          euler="0 -0.3 0" rgba="0.3 0.6 0.6 1"/>
    
    <!-- Stairs on the sides -->
    <geom name="left_step1" type="box" pos="4 -2.0 0.075" size="0.3 0.3 0.075" rgba="0.6 0.4 0.2 1"/>
    <geom name="left_step2" type="box" pos="4.6 -2.0 0.225" size="0.3 0.3 0.075" rgba="0.6 0.4 0.2 1"/>
    
    <geom name="right_step1" type="box" pos="6 2.0 0.075" size="0.3 0.3 0.075" rgba="0.6 0.4 0.2 1"/>
    <geom name="right_step2" type="box" pos="6.6 2.0 0.225" size="0.3 0.3 0.075" rgba="0.6 0.4 0.2 1"/>
  </worldbody>'''

modified_xml = xml_content.replace('</worldbody>', fun_scene_xml)

print("Loading G1 in fun colorful environment...")
model = mujoco.MjModel.from_xml_string(modified_xml)
data = mujoco.MjData(model)
model.opt.timestep = SIMULATION_DT

# Initialize robot
mujoco.mj_resetData(model, data)
data.qpos[7:] = DEFAULT_ANGLES
data.qpos[2] = 0.77
mujoco.mj_forward(model, data)

# Load policy
print("Loading walking policy...")
policy = torch.jit.load(POLICY_PATH)
print("✅ Ready!")

os.chdir(original_dir)

# Initialize policy variables
action = np.zeros(NUM_ACTIONS, dtype=np.float32)
target_dof_pos = DEFAULT_ANGLES.copy()
obs = np.zeros(NUM_OBS, dtype=np.float32)
counter = 0

# Create camera renderer
WIDTH, HEIGHT = 640, 480
renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

# Create video writer
video_path = os.path.join(RESULTS_DIR, "robot_walking_fun_scene.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 60
video_writer = cv2.VideoWriter(video_path, fourcc, fps, (WIDTH, HEIGHT))

# Create camera
camera = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, camera)

# Create display window
cv2.namedWindow("Robot's Perspective", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Robot's Perspective", WIDTH, HEIGHT)

print(f"\n🎬 Recording 20-second video...")
print(f"🎥 Watch both windows:")
print(f"   - MuJoCo Viewer: See the robot in the colorful world")
print(f"   - Robot's Perspective: What the robot sees\n")

max_frames = 600  # 20 seconds at 30fps

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    
    while viewer.is_running() and counter < max_frames * CONTROL_DECIMATION:
        step_start = time.time()
        
        # Apply control
        tau = pd_control(target_dof_pos, data.qpos[7:], KPS, np.zeros_like(KDS), data.qvel[6:], KDS)
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        counter += 1
        
        # Policy inference
        if counter % CONTROL_DECIMATION == 0:
            qj = data.qpos[7:]
            dqj = data.qvel[6:]
            quat = data.qpos[3:7]
            omega = data.qvel[3:6]
            
            qj_scaled = (qj - DEFAULT_ANGLES) * DOF_POS_SCALE
            dqj_scaled = dqj * DOF_VEL_SCALE
            gravity_orientation = get_gravity_orientation(quat)
            omega_scaled = omega * ANG_VEL_SCALE
            
            period = 0.8
            count = counter * SIMULATION_DT
            phase = (count % period) / period
            sin_phase = np.sin(2 * np.pi * phase)
            cos_phase = np.cos(2 * np.pi * phase)
            
            obs[:3] = omega_scaled
            obs[3:6] = gravity_orientation
            obs[6:9] = CMD * CMD_SCALE
            obs[9:9+NUM_ACTIONS] = qj_scaled
            obs[9+NUM_ACTIONS:9+2*NUM_ACTIONS] = dqj_scaled
            obs[9+2*NUM_ACTIONS:9+3*NUM_ACTIONS] = action
            obs[9+3*NUM_ACTIONS:9+3*NUM_ACTIONS+2] = np.array([sin_phase, cos_phase])
            
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            action = policy(obs_tensor).detach().numpy().squeeze()
            target_dof_pos = action * ACTION_SCALE + DEFAULT_ANGLES
        
        # Render camera view every few steps (for 30fps video)
        if counter % (CONTROL_DECIMATION // 3) == 0:
            robot_pos = data.qpos[0:3]
            head_height = 0.5
            
            # Camera positioned at robot's head, looking straight forward
            camera.lookat[:] = [robot_pos[0] + 3, robot_pos[1], robot_pos[2] + head_height]  # Look further ahead
            camera.distance = head_height + 0.3  # Slightly behind head
            camera.azimuth = 0  # Straight forward
            camera.elevation = -5  # Slight downward angle
            
            renderer.update_scene(data, camera=camera)
            pixels = renderer.render()
            
            frame_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
            
            cv2.imshow("Robot's Perspective", frame_bgr)
            cv2.waitKey(1)
        
        viewer.sync()
        
        # Progress update
        if counter % 500 == 0:
            elapsed = time.time() - start_time
            robot_x = data.qpos[0]
            print(f"📹 {elapsed:.1f}s - Robot at X={robot_x:.2f}m")
        
        # Time keeping
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

video_writer.release()
cv2.destroyAllWindows()

print(f"\n✅ Video saved: {video_path}")
print(f"🎬 Duration: 20 seconds")
print(f"🎨 Robot walking through colorful environment!")
print(f"👀 Shows what robot sees as it walks!")