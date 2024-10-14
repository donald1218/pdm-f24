import numpy as np
from PIL import Image
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import os
import sys
import argparse
import shutil
from map import get_path,get_map
from scipy.spatial.transform import Rotation as R
from make_gif import make_gif
import json
import pandas as pd

# This is the scene we are going to load.
# support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
### put your scene path ###
test_scene = "replica_v1/apartment_0/habitat/mesh_semantic.ply"

sim_settings = {
    "scene": test_scene,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
    "width": 512,  # Spatial resolution of the observations
    "height": 512,
    "sensor_pitch": 0,  # sensor pitch (x rotation in rads)
}


# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def transform_semantic(semantic_obs):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
    return semantic_img

def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=0.1)),
        "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=1.0)),
        "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=1.0)),
    }

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #depth snesor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #semantic snesor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def navigateAndSee(action="", data_root='data_collection/first_floor/',target_id = 0,color_code = [0,0,0]):
    global count
    observations = sim.step(action)
    #print("action: ", action)

    agent_state = agent.get_state()
    sensor_state = agent_state.sensor_states['color_sensor']
    print("Frame:", count)
    print("camera pose: x y z rw rx ry rz")
    print(sensor_state.position[0],sensor_state.position[1],sensor_state.position[2], sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z)
    
    count += 1
    rgb_img = transform_rgb_bgr(observations["color_sensor"])
    
    with open("./replica_v1/apartment_0/habitat/info_semantic.json", "r") as f:
        annotations = json.load(f)
    instance_id_to_semantic_label_id = np.array(annotations["id_to_label"])
    semantic_id = instance_id_to_semantic_label_id[observations["semantic_sensor"]]
    
    red_mask = np.full(rgb_img.shape, (0,0,255), dtype=np.uint8)
    target_id_region = np.where(semantic_id == target_id + 1) 

    
    if len(target_id_region[0]) != 0:
        blend_img = cv2.addWeighted(rgb_img, 0.5, red_mask, 0.5, 0)
        rgb_img[target_id_region] = blend_img[target_id_region]

    
    cv2.imwrite(data_root + f"rgb/{count}.png", rgb_img)

    
    
def get_action(now):
    agent_state = agent.get_state()
    sensor_state = agent_state.sensor_states['color_sensor']
    if now < len(path)-1:
        if np.sqrt(((sensor_state.position[0] - path[now+1][0] ) **2 + (sensor_state.position[2] - path[now+1][1])**2)) < 0.1:
            now += 1 
    now_degree = np.round(np.degrees(2 * np.arctan2(sensor_state.rotation.y, sensor_state.rotation.w)))

    vector = [sensor_state.position[0] - path[now+1][0] , sensor_state.position[2] - path[now+1][1] ]
    angle = np.arctan2(vector[0],vector[1]) 
    
    
    angle_d = np.round(np.degrees(angle))
    if now_degree < 0:
        now_degree = now_degree + 360
    print("now_degree: ", now_degree)
        
    if angle_d < 0:
        angle_d = angle_d + 360
        
    rotate_angle = angle_d - now_degree
        
    print("angle_d: ", angle_d)
    print("now", now)
    if rotate_angle == 0:
        if now == len(path) -2 :
            return now,"finish"
        return now,"move_forward"   
    elif rotate_angle > 0 and rotate_angle < 180:
        return now,"turn_left"
    return now,"turn_right"
    

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--floor', type=int, default=1)  
args = parser.parse_args()

get_map()
map = cv2.imread('map.png')
target_object,target_object_id,color_code,path = get_path(map)
print(path)

cfg = make_simple_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)

# initialize an agent
agent = sim.initialize_agent(sim_settings["default_agent"])

# Set agent state
agent_state = habitat_sim.AgentState()
if args.floor == 1:
    agent_state.position = np.array([path[0][0], 0.0, path[0][1]])  # agent in world space
    
agent.set_state(agent_state)


# obtain the default, discrete actions that an agent can perform
# default action space contains 3 actions: move_forward, turn_left, and turn_right
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Discrete action space: ", action_names)


if args.floor == 1:
    data_root = "data_collection/first_floor/"


if os.path.isdir(data_root): 
    shutil.rmtree(data_root)  # WARNING: this line will delete whole directory with files

for sub_dir in ['rgb/']:
    os.makedirs(data_root + sub_dir)

count = 0
action = "move_forward"

navigateAndSee(action, data_root,target_object_id,color_code)
now = 0 
while True:
    # keystroke = cv2.waitKey(0)
    now,action = get_action(now)
    if action == "move_forward":
        print("action: FORWARD")
    elif action == "turn_left" :
        print("action: LEFT")
    elif action == "turn_right":
        print("action: RIGHT")
    elif action == "finish":
        print("action: FINISH")
        break
    navigateAndSee(action, data_root,target_object_id,color_code)

make_gif(target_object)

