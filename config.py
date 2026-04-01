import os

_BASE_DIR = os.path.dirname(__file__)

# Robot assets — all needed files live in assets/g1/
ROBOT_DIR   = os.path.join(_BASE_DIR, "assets", "g1")
POLICY_PATH = os.path.join(ROBOT_DIR, "motion.pt")
SCENE_DIR   = os.path.join(_BASE_DIR, "scenes")
