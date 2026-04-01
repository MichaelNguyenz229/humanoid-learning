"""
terrain_generator.py — Generate terrain and preview it immediately

Edit the __main__ block below to configure your terrain, then run:
    mjpython terrain_generator.py

Workflow:
    1. Enter a scene name when prompted
    2. Terrain is generated and saved to scenes/{name}.xml
    3. Viewer launches automatically showing terrain + static robot at start position
"""

import xml.etree.ElementTree as xml_et
import numpy as np
import cv2
import noise
import os
import sys
import mujoco
import mujoco.viewer

# ── Paths ─────────────────────────────────────────────────────────────────────
SCENE_DIR        = "scenes/"
INPUT_SCENE_PATH = os.path.join(SCENE_DIR, "flat.xml")
ROBOT            = "g1"

name             = input("Enter scene name (e.g. stairs): ").strip()
OUTPUT_SCENE_PATH = os.path.join(SCENE_DIR, f"{name}.xml")
os.makedirs(SCENE_DIR, exist_ok=True)


# ── Math helpers ──────────────────────────────────────────────────────────────

def euler_to_quat(roll, pitch, yaw):
    cx, sx = np.cos(roll / 2),  np.sin(roll / 2)
    cy, sy = np.cos(pitch / 2), np.sin(pitch / 2)
    cz, sz = np.cos(yaw / 2),   np.sin(yaw / 2)
    return np.array([
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    ], dtype=np.float64)

def euler_to_rot(roll, pitch, yaw):
    rot_x = np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
    rot_y = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
    rot_z = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    return rot_z @ rot_y @ rot_x

def rot2d(x, y, yaw):
    return x * np.cos(yaw) - y * np.sin(yaw), x * np.sin(yaw) + y * np.cos(yaw)

def rot3d(pos, euler):
    return euler_to_rot(euler[0], euler[1], euler[2]) @ pos

def list_to_str(vec):
    return " ".join(str(s) for s in vec)


# ── Terrain generator ─────────────────────────────────────────────────────────

class TerrainGenerator:

    def __init__(self):
        self.scene    = xml_et.parse(INPUT_SCENE_PATH)
        self.root     = self.scene.getroot()
        self.worldbody = self.root.find("worldbody")
        self.asset    = self.root.find("asset")

    def AddBox(self, position=[1,0,0], euler=[0,0,0], size=[0.1,0.1,0.1]):
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["pos"]  = list_to_str(position)
        geo.attrib["type"] = "box"
        geo.attrib["size"] = list_to_str(0.5 * np.array(size))
        geo.attrib["quat"] = list_to_str(euler_to_quat(*euler))

    def AddGeometry(self, position=[1,0,0], euler=[0,0,0], size=[0.1,0.1], geo_type="box"):
        # geo_type: "plane", "sphere", "capsule", "ellipsoid", "cylinder", "box"
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["pos"]  = list_to_str(position)
        geo.attrib["type"] = geo_type
        geo.attrib["size"] = list_to_str(0.5 * np.array(size))
        geo.attrib["quat"] = list_to_str(euler_to_quat(*euler))

    def AddStairs(self, init_pos=[1,0,0], yaw=0.0, width=0.2,
                  height=0.15, length=1.5, stair_nums=10):
        local_pos = [0.0, 0.0, -0.5 * height]
        for i in range(stair_nums):
            local_pos[0] += width
            local_pos[2] += height
            x, y = rot2d(local_pos[0], local_pos[1], yaw)
            self.AddBox([x + init_pos[0], y + init_pos[1], local_pos[2]],
                        [0.0, 0.0, yaw], [width, length, height])

    def AddSuspendStairs(self, init_pos=[1,0,0], yaw=1, width=0.2,
                         height=0.05, length=3, gap=0.1, stair_nums=30):
        local_pos = [0.0, 0.0, -0.5 * height]
        for i in range(stair_nums):
            local_pos[0] += width
            local_pos[2] += height
            x, y = rot2d(local_pos[0], local_pos[1], yaw)
            self.AddBox([x + init_pos[0], y + init_pos[1], local_pos[2]],
                        [0.0, 0.0, yaw], [width, length, abs(height - gap)])

    def AddRoughGround(self, init_pos=[1,0,0], euler=[0,0,0], nums=[10,10],
                       box_size=[.5,.5,0.5], box_euler=[0,0,0],
                       separation=[0.2,0.2], box_size_rand=[0.05,0.05,0.05],
                       box_euler_rand=[0.2,0.2,0.2], separation_rand=[0.05,0.05]):
        local_pos = [0.0, 0.0, -0.5 * box_size[2]]
        new_separation = np.array(separation) + np.array(separation_rand) * np.random.uniform(-1, 1, 2)
        for i in range(nums[0]):
            local_pos[0] += new_separation[0]
            local_pos[1] = 0.0
            for j in range(nums[1]):
                new_box_size  = np.array(box_size)  + np.array(box_size_rand)  * np.random.uniform(-1, 1, 3)
                new_box_euler = np.array(box_euler) + np.array(box_euler_rand) * np.random.uniform(-1, 1, 3)
                new_separation = np.array(separation) + np.array(separation_rand) * np.random.uniform(-1, 1, 2)
                local_pos[1] += new_separation[1]
                pos = rot3d(local_pos, euler) + np.array(init_pos)
                self.AddBox(pos, new_box_euler, new_box_size)

    def AddPerlinHeighField(self, position=[1,0,0], euler=[0,0,0],
                            size=[1.0,1.0], height_scale=0.4, negative_height=0.4,
                            image_width=128, img_height=128, smooth=40,
                            perlin_octaves=90, perlin_persistence=0.5,
                            perlin_lacunarity=2.0, output_hfield_image="height_field.png"):
        terrain_image = np.zeros((img_height, image_width), dtype=np.uint8)
        for y in range(image_width):
            for x in range(image_width):
                noise_value = noise.pnoise2(x / smooth, y / smooth,
                                            octaves=perlin_octaves,
                                            persistence=perlin_persistence,
                                            lacunarity=perlin_lacunarity)
                terrain_image[y, x] = int((noise_value + 1) / 2 * 255)
        cv2.imwrite(os.path.join(SCENE_DIR, output_hfield_image), terrain_image)
        hfield = xml_et.SubElement(self.asset, "hfield")
        hfield.attrib["name"] = "perlin_hfield"
        hfield.attrib["size"] = list_to_str([size[0]/2, size[1]/2, height_scale, negative_height])
        hfield.attrib["file"] = output_hfield_image
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["type"]   = "hfield"
        geo.attrib["hfield"] = "perlin_hfield"
        geo.attrib["pos"]    = list_to_str(position)
        geo.attrib["quat"]   = list_to_str(euler_to_quat(*euler))

    def AddHeighFieldFromImage(self, position=[1,0,0], euler=[0,0,0],
                               size=[2.0,1.6], height_scale=0.02, negative_height=0.1,
                               input_img=None, output_hfield_image="height_field.png",
                               image_scale=[1.0,1.0], invert_gray=False):
        input_image   = cv2.imread(input_img)
        width         = int(input_image.shape[1] * image_scale[0])
        height        = int(input_image.shape[0] * image_scale[1])
        resized_image = cv2.resize(input_image, (width, height), interpolation=cv2.INTER_AREA)
        terrain_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        if invert_gray:
            terrain_image = 255 - terrain_image
        cv2.imwrite(os.path.join(SCENE_DIR, output_hfield_image), terrain_image)
        hfield = xml_et.SubElement(self.asset, "hfield")
        hfield.attrib["name"] = "image_hfield"
        hfield.attrib["size"] = list_to_str([size[0]/2, size[1]/2, height_scale, negative_height])
        hfield.attrib["file"] = output_hfield_image
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["type"]   = "hfield"
        geo.attrib["hfield"] = "image_hfield"
        geo.attrib["pos"]    = list_to_str(position)
        geo.attrib["quat"]   = list_to_str(euler_to_quat(*euler))

    def Save(self):
        self.scene.write(OUTPUT_SCENE_PATH)
        print(f"Saved → {OUTPUT_SCENE_PATH}")


# ── Viewer ────────────────────────────────────────────────────────────────────

def launch_viewer(scene_path):
    """Load generated scene with static robot at spawn position and open viewer."""
    print(f"Launching viewer for: {scene_path}")
    print("Mouse drag to rotate, scroll to zoom, right-click drag to pan, ESC to exit\n")

    model = mujoco.MjModel.from_xml_path(os.path.abspath(scene_path))
    data  = mujoco.MjData(model)

    # Compute initial pose without stepping physics
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type     = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.distance = 6.0
        viewer.cam.elevation = -25
        viewer.cam.azimuth   = 45

        while viewer.is_running():
            # No physics — just hold viewer open for inspection
            mujoco.mj_forward(model, data)
            viewer.sync()


# ── Main — edit terrain config here ──────────────────────────────────────────

if __name__ == "__main__":
    tg = TerrainGenerator()

    # ── Uncomment and edit whichever terrain you want ─────────────────────────

    # Box obstacle
    # tg.AddBox(position=[1.5, 0.0, 0.1], euler=[0, 0, 0.0], size=[1, 1.5, 0.2])

    # Geometry obstacle
    # geo_type: "plane", "sphere", "capsule", "ellipsoid", "cylinder", "box"
    # tg.AddGeometry(position=[1.5, 0.0, 0.25], euler=[0,0,0], size=[1.0,0.5,0.5], geo_type="cylinder")

    # Slope
    # level 1: position=[5.3, 0, 0.45],  euler=[0.0, -0.1, 0.0], size=[10, 10, 0.1]
    # level 2: position=[5.3, 0, 1.3],   euler=[0.0, -0.3, 0.0], size=[10, 10, 0.1]
    # level 3: position=[5.3, 0, 1.8],   euler=[0.0, -0.4, 0.0], size=[10, 10, 0.1]
    tg.AddBox(position=[.6, 0, 0.0], euler=[0.0, -0.15, 0.0], size=[2, 15, 0.1])

    # Stairs
    # tg.AddStairs(init_pos=[1.0, 0.0, 0.0], yaw=0.0)

    # Suspend stairs
    # tg.AddSuspendStairs(init_pos=[1.0, 0, 0.0], yaw=0.0)

    # Rough ground
    # tg.AddRoughGround(init_pos=[.5, -5, 0.03], euler=[0, 0, 0.0], nums=[50, 50])

    # Perlin height field
    tg.AddPerlinHeighField(position=[11.5, 0.0, -0.02], size=[20, 15])

    # Height field from image
    # tg.AddHeighFieldFromImage(position=[-1.5, 2.0, 0.0], euler=[0, 0, -1.57],
    #                           size=[2.0, 2.0], input_img="./unitree_robot.jpeg")

    # ── Save and preview ──────────────────────────────────────────────────────
    tg.Save()
    launch_viewer(OUTPUT_SCENE_PATH)