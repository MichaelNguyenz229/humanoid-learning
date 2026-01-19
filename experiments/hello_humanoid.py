import mujoco
import mujoco.viewer
import numpy as np

# Load the humanoid model (this comes built-in with MuJoCo)
model = mujoco.MjModel.from_xml_string("""
<mujoco>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="sphere" size=".1" rgba="0 .9 0 1"/>
    </body>
  </worldbody>
</mujoco>
""")
data = mujoco.MjData(model)

print(f"🤖 Humanoid loaded!")
print(f"   - Number of joints: {model.nv}")
print(f"   - Number of actuators: {model.nu}")

# Launch the interactive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Step the physics simulation forward
        mujoco.mj_step(model, data)
        
        # Sync the viewer with the simulation
        viewer.sync()