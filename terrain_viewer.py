"""
terrain_viewer.py — View terrain scenes without the robot

Edit the config block below, then run:
    mjpython terrain_viewer.py
"""

import mujoco
import mujoco.viewer
import os
import sys

# ── Edit these ────────────────────────────────────────────────────────────────

SCENE = input("Scene (flat | stairs1 | stairs2 | ...): ").strip()

# ─────────────────────────────────────────────────────────────────────────────

scene_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes", f"{SCENE}.xml")

if not os.path.exists(scene_path):
    print(f"Error: scene not found at {scene_path}")
    sys.exit(1)

print(f"\nLoading: {scene_path}")

# ── Strip robot from scene — load terrain geometry only ───────────────────────
# We load the XML as a string and remove the robot include so only
# the worldbody terrain geoms are loaded, no joints or actuators
with open(scene_path, "r") as f:
    xml = f.read()

# Remove any <include> tags (robot body definitions)
import re
xml = re.sub(r'<include\s+[^/]*/>', '', xml)

# Remove <actuator> block if present
xml = re.sub(r'<actuator>.*?</actuator>', '', xml, flags=re.DOTALL)

# Remove any <body> tags that contain freejoint (the robot root body)
xml = re.sub(r'<body[^>]*>.*?<freejoint[^/]*/?>.*?</body>', '', xml, flags=re.DOTALL)

try:
    model = mujoco.MjModel.from_xml_string(xml)
except Exception as e:
    # If stripping fails just load the full scene — robot will be there but static
    print(f"Note: could not strip robot ({e}), loading full scene")
    model = mujoco.MjModel.from_xml_path(scene_path)

data = mujoco.MjData(model)

print("Controls: mouse drag to rotate, scroll to zoom, right-click drag to pan")
print("Press ESC to exit\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Start with a good overview angle
    viewer.cam.type     = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.distance = 6.0
    viewer.cam.elevation = -30
    viewer.cam.azimuth   = 45

    while viewer.is_running():
        # No physics needed — just keep viewer alive
        mujoco.mj_forward(model, data)
        viewer.sync()