import mujoco
import mujoco.viewer
import os
import numpy as np

# Path to Unitree G1 model
#model_path = os.path.join(PROJECT_ROOT,"models", "unitree_rl_gym", "resources", "robots", "g1_description", "scene.xml")

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROBOT_DIR

#path with terrain
model_path = os.path.join(ROBOT_DIR, "scene.xml")
# Load the G1 humanoid

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

print(f"🤖 Unitree G1 Humanoid loaded!")
print(f"   - Number of joints: {model.nv}")
print(f"   - Number of actuators: {model.nu}")

# Store the initial standing position
mujoco.mj_resetDataKeyframe(model, data, 0)  # Reset to default pose
initial_qpos = data.qpos.copy()

print("\n🎮 Controls:")
print("   - Try clicking/dragging the robot")
print("   - Watch it try to recover!")
print("   - Press ESC to exit\n")

# Simple PD controller gains
kp = 100  # Position gain
kd = 5   # Velocity gain

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # PD control: try to return to initial pose
        data.ctrl[:] = kp * (initial_qpos[7:] - data.qpos[7:]) - kd * data.qvel[6:]
        
        # Step physics
        mujoco.mj_step(model, data)
        viewer.sync()