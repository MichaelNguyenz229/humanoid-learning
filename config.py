import os

# Each dev sets their own modesl path
MODELS_DIR = os.environ.get("HUMANOID_MODELS_DIR", os.path.join(os.path.dirname(__file__), "models")) #__file__ is a special python var, full path to this script
ROBOT_DIR = os.path.join(MODELS_DIR, "unitree_rl_gym", "resources", "robots", "g1_description")
POLICY_PATH = os.path.join(MODELS_DIR,"unitree_rl_gym", "deploy", "pre_train", "g1","motion.pt")
SCENE_DIR = os.path.join(os.path.dirname(__file__), "tests", "scenes")

